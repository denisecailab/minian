import os
import re
import shutil
import warnings
from copy import deepcopy
from os import listdir
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path
from uuid import uuid4

import ffmpeg
import cv2
import dask as da
import dask.array as darr
import numpy as np
import pandas as pd
import psutil
import rechunker
import xarray as xr
import zarr as zr
from dask.optimization import inline_functions
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import SchedulerState, cast
from natsort import natsorted
from tifffile import TiffFile, imread


def load_videos(
    vpath,
    pattern=r"msCam[0-9]+\.avi$",
    dtype=np.float64,
    downsample=None,
    downsample_strategy="subset",
    post_process=None,
):
    """
    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together across time and return a
    `xarray.DataArray` representation of the concatenated videos. The default
    assumption is video filenames start with ``msCam`` followed by at least a
    number, and then followed by ``.avi``. In addition, it is assumed that the
    name of the folder correspond to a recording session identifier.

    Parameters
    ----------
    vpath : str
        The path to search for videos
    pattern : str, optional
        The pattern that describes filenames of videos. (Default value =
        'msCam[0-9]+\.avi')

    Returns
    -------
    xarray.DataArray or None
        The labeled 3-d array representation of the videos with dimensions:
        ``frame``, ``height`` and ``width``. Returns ``None`` if no data was
        found in the specified folder.
    """
    vpath = os.path.normpath(vpath)
    ssname = os.path.basename(vpath)
    vlist = natsorted(
        [vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)]
    )
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath)
        )
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension in (".avi", ".mkv"):
        movie_load_func = load_avi_lazy
    elif file_extension == ".tif":
        movie_load_func = load_tif_lazy
    else:
        raise ValueError("Extension not supported.")

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr,
        dims=["frame", "height", "width"],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2]),
        ),
    )
    if dtype:
        varr = varr.astype(dtype)
    if downsample:
        if downsample_strategy == "mean":
            varr = varr.coarsen(**downsample, boundary="trim", coord_func="min").mean()
        elif downsample_strategy == "subset":
            varr = varr.isel(**{d: slice(None, None, w) for d, w in downsample.items()})
        else:
            raise NotImplementedError("unrecognized downsampling strategy")
    varr = varr.rename("fluorescence")
    if post_process:
        varr = post_process(varr, vpath, ssname, vlist, varr_list)
    return varr


def load_tif_lazy(fname):
    data = TiffFile(fname)
    f = len(data.pages)

    fmread = da.delayed(load_tif_perframe)
    flist = [fmread(fname, i) for i in range(f)]

    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_tif_perframe(fname, fid):
    return imread(fname, key=fid)


def load_avi_lazy_framewise(fname):
    cap = cv2.VideoCapture(fname)
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fmread = da.delayed(load_avi_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [
        da.array.from_delayed(fm, dtype=sample.dtype, shape=sample.shape)
        for fm in flist
    ]
    return da.array.stack(arr, axis=0)


def load_avi_lazy(fname):
    probe = ffmpeg.probe(fname)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w = int(video_info["width"])
    h = int(video_info["height"])
    f = int(video_info["nb_frames"])
    return da.array.from_delayed(
        da.delayed(load_avi_ffmpeg)(fname, h, w, f), dtype=np.uint8, shape=(f, h, w)
    )


def load_avi_ffmpeg(fname, h, w, f):
    out_bytes, err = (
        ffmpeg.input(fname)
        .video.output("pipe:", format="rawvideo", pix_fmt="gray")
        .run(capture_stdout=True)
    )
    return np.frombuffer(out_bytes, np.uint8).reshape(f, h, w)


def load_avi_perframe(fname, fid):
    cap = cv2.VideoCapture(fname)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, fm = cap.read()
    if ret:
        return np.flip(cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY), axis=0)
    else:
        print("frame read failed for frame {}".format(fid))
        return np.zeros((h, w))


def open_minian(dpath, post_process=None, return_dict=False):
    """
    Opens a file previously saved in minian handling the proper data format and chunks

    Args:
        dpath ([string]): contains the normalized absolutized version of the pathname path,which is the path to minian folder;
        Post_process (function): post processing function, parameters: dataset (xarray.DataArray), mpath (string, path to the raw backend files)
        return_dict ([boolean]): default False

    Returns:
        xarray.DataArray: [loaded data]
    """
    dslist = []
    for d in listdir(dpath):
        arr_path = pjoin(dpath, d)
        if isdir(arr_path):
            arr = list(xr.open_zarr(arr_path).values())[0]
            arr.data = darr.from_zarr(
                os.path.join(arr_path, arr.name), inline_array=True
            )
            dslist.append(arr)
    if return_dict:
        ds = {d.name: d for d in dslist}
    else:
        ds = xr.merge(dslist, compat="no_conflicts")
    if (not return_dict) and post_process:
        ds = post_process(ds, dpath)
    return ds


def open_minian_mf(
    dpath,
    index_dims,
    result_format="xarray",
    pattern=r"minian\.[0-9]+$",
    sub_dirs=[],
    exclude=True,
    **kwargs
):
    minian_dict = dict()
    for nextdir, dirlist, filelist in os.walk(dpath, topdown=False):
        nextdir = os.path.abspath(nextdir)
        cur_path = Path(nextdir)
        dir_tag = bool(
            (
                (any([Path(epath) in cur_path.parents for epath in sub_dirs]))
                or nextdir in sub_dirs
            )
        )
        if exclude == dir_tag:
            continue
        flist = list(filter(lambda f: re.search(pattern, f), filelist + dirlist))
        if flist:
            print("opening dataset under {}".format(nextdir))
            if len(flist) > 1:
                warnings.warn("multiple dataset found: {}".format(flist))
            fname = flist[-1]
            print("opening {}".format(fname))
            minian = open_minian(dpath=os.path.join(nextdir, fname), **kwargs)
            key = tuple([np.array_str(minian[d].values) for d in index_dims])
            minian_dict[key] = minian
            print(["{}: {}".format(d, v) for d, v in zip(index_dims, key)])

    if result_format == "xarray":
        return xrconcat_recursive(minian_dict, index_dims)
    elif result_format == "pandas":
        minian_df = pd.Series(minian_dict).rename("minian")
        minian_df.index.set_names(index_dims, inplace=True)
        return minian_df.to_frame()
    else:
        raise NotImplementedError("format {} not understood".format(result_format))


def save_minian(
    var, dpath, meta_dict=None, overwrite=False, chunks=None, mem_limit="200MB"
):
    """
    Saves the data (var) in the format specified by the backend variable, in the location specified by dpath under the name ‘minian’, if overwrite True
    Args:
        var (xarray.DataArray): data to be saved
        dpath (str): path where to save the data
        fname (str, optional): output file name. Defaults to 'minian'.
        backend (str, optional): file storage format. Defaults to 'netcdf'.
        meta_dict (dict, optional): metadata for example {‘animal’: -3, ‘session’: -2, ‘session_id’: -1}. Key value pair. Defaults to None.
        overwrite (bool, optional): if true overwrites a file in the same location with the same name. Defaults to False.

    Raises:
        NotImplementedError

    Returns:
        xarray.DataArray: the saved var xarray.DataArray
    """
    dpath = os.path.normpath(dpath)
    Path(dpath).mkdir(parents=True, exist_ok=True)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.abspath(dpath).split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
        )
    md = {True: "a", False: "w-"}[overwrite]
    fp = os.path.join(dpath, var.name + ".zarr")
    if overwrite:
        try:
            shutil.rmtree(fp)
        except FileNotFoundError:
            pass
    ds.to_zarr(fp, mode=md)
    if chunks is not None:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))
        zstore = zr.open(fp)
        rechk = rechunker.rechunk(
            zstore[var.name], chunks, mem_limit, dst_path, temp_store=temp_path
        )
        rechk.execute()
        try:
            shutil.rmtree(temp_path)
        except FileNotFoundError:
            pass
        arr_path = os.path.join(fp, var.name)
        for f in os.listdir(arr_path):
            os.remove(os.path.join(arr_path, f))
        for f in os.listdir(dst_path):
            os.rename(os.path.join(dst_path, f), os.path.join(arr_path, f))
        os.rmdir(dst_path)
    arr = xr.open_zarr(fp)[var.name]
    arr.data = darr.from_zarr(os.path.join(fp, var.name), inline_array=True)
    return arr


def xrconcat_recursive(var, dims):
    if len(dims) > 1:
        if type(var) is dict:
            var_dict = var
        elif type(var) is list:
            var_dict = {tuple([np.asscalar(v[d]) for d in dims]): v for v in var}
        else:
            raise NotImplementedError("type {} not supported".format(type(var)))
        try:
            var_dict = {k: v.to_dataset() for k, v in var_dict.items()}
        except AttributeError:
            pass
        var_ps = pd.Series(var_dict)
        var_ps.index.set_names(dims, inplace=True)
        xr_ls = []
        for idx, v in var_ps.groupby(level=dims[0]):
            v.index = v.index.droplevel(dims[0])
            xarr = xrconcat_recursive(v.to_dict(), dims[1:])
            xr_ls.append(xarr)
        return xr.concat(xr_ls, dim=dims[0])
    else:
        if type(var) is dict:
            var = var.values()
        return xr.concat(var, dim=dims[0])


def update_meta(dpath, pattern=r"^minian\.nc$", meta_dict=None, backend="netcdf"):
    for dirpath, dirnames, fnames in os.walk(dpath):
        if backend == "netcdf":
            fnames = filter(lambda fn: re.search(pattern, fn), fnames)
        elif backend == "zarr":
            fnames = filter(lambda fn: re.search(pattern, fn), dirnames)
        else:
            raise NotImplementedError("backend {} not supported".format(backend))
        for fname in fnames:
            f_path = os.path.join(dirpath, fname)
            pathlist = os.path.normpath(dirpath).split(os.sep)
            new_ds = xr.Dataset()
            old_ds = open_minian(f_path, f_path, backend)
            new_ds.attrs = deepcopy(old_ds.attrs)
            old_ds.close()
            new_ds = new_ds.assign_coords(
                **dict(
                    [(cdname, pathlist[cdval]) for cdname, cdval in meta_dict.items()]
                )
            )
            if backend == "netcdf":
                new_ds.to_netcdf(f_path, mode="a")
            elif backend == "zarr":
                new_ds.to_zarr(f_path, mode="w")
            print("updated: {}".format(f_path))


def get_chk(arr):
    return {d: c for d, c in zip(arr.dims, arr.chunks)}


def rechunk_like(x, y):
    """
    Resizes chunks based on the new input dimensions

    Args:
        x (array): the array to be rechunked. i.e. destination of rechunking
        y (array): the array where chunk information are extracted. i.e. the source of rechunking

    Returns:
        dict: data with new dimensions as specified in the input
    """
    try:
        dst_chk = get_chk(y)
        comm_dim = set(x.dims).intersection(set(dst_chk.keys()))
        dst_chk = {d: max(dst_chk[d]) for d in comm_dim}
        return x.chunk(dst_chk)
    except TypeError:
        return x.compute()


def get_optimal_chk(ref, arr=None, dim_grp=None, ncores="auto", mem_limit="auto"):
    """
    Estimates the chunk of video (i.e. video sizes and number of frames) that optimizes computer memory use when the script is run parallel over multiple cores.

    Args:
        ref (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        arr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width. Defaults to None.
        dim_grp (array, optional): provide labels for the dimension of the data. Defaults to None.
        ncores (str, optional): number of CPU cores. Defaults to 'auto'.
        mem_limit (str, optional): max available memory that can be given to a process without the system going into swap. Defaults to 'auto'.

    Returns:
        dict: sizes of the chunks that optimize memory usage in parallel computing the key is the dimension, the value is the max chunk size
    """
    if arr is None:
        arr = ref
    szs = ref.sizes
    if ncores == "auto":
        ncores = psutil.cpu_count()
    if mem_limit == "auto":
        mem_limit = psutil.virtual_memory().available / (1024 ** 2)
    tempsz = (
        1000
        * (3 * szs["height"] * szs["width"] + 7 * szs["frame"])
        * ref.dtype.itemsize
        / (1024 ** 2)
    )
    csize = int(np.floor((mem_limit - tempsz) / ncores / 4))
    if csize < 64:
        warnings.warn(
            "estimated memory limit is smaller than 64MiB. Using 64MiB chunksize instead. "
        )
        csize = 64
    csize = int(np.floor((mem_limit) / ncores))
    if csize > 512:
        warnings.warn(
            "estimated memory limit is bigger than 512MiB. Using 512MiB chunksize instead. "
        )
        csize = 512
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]
    opt_chk = dict()
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: "auto" for d in dg}
        dr_dict = {d: -1 for d in d_rest}
        dg_dict.update(dr_dict)
        with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
            arr_chk = arr.chunk(dg_dict)
        re_dict = {d: c for d, c in zip(dims, arr_chk.chunks)}
        re_dict = {d: max(re_dict[d]) for d in dg}
        opt_chk.update(re_dict)
    return opt_chk


class TaskAnnotation(SchedulerPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.annt_dict = {
            "load_avi_ffmpeg": {"resources": {"MEM": 1}, "priority": -1},
            "est_sh_chunk": {"resources": {"MEM": 0.5}},
        }

    def update_graph(self, scheduler, client, tasks, **kwargs):
        parent = cast(SchedulerState, scheduler)
        # set_trace()
        for tk in tasks.keys():
            for pattern, annt in self.annt_dict.items():
                if re.search(pattern, tk):
                    ts = parent._tasks.get(tk)
                    res = annt.get("resources", None)
                    if res:
                        ts._resource_restrictions = res
                    pri = annt.get("priority", None)
                    if pri:
                        pri_org = list(ts._priority)
                        pri_org[0] = -pri
                        ts._priority = tuple(pri_org)


def inline_zarr(dsk, keys):
    dsk = inline_functions(
        dsk,
        [],
        fast_functions=[darr.core.getter, zr.core.Array],
        inline_constants=True,
    )
    return darr.optimization.optimize(dsk, keys)
