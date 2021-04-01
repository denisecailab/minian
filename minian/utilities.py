import functools as fct
import os
import re
import shutil
import warnings
from copy import deepcopy
from os import listdir
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path
from typing import Callable, List, Optional, Union
from uuid import uuid4

import _operator
import cv2
import dask as da
import dask.array as darr
import ffmpeg
import numpy as np
import pandas as pd
import rechunker
import xarray as xr
import zarr as zr
from dask.core import flatten
from dask.delayed import optimize as default_delay_optimize
from dask.optimization import cull, fuse, inline, inline_functions
from dask.utils import ensure_dict
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import SchedulerState, cast
from natsort import natsorted
from tifffile import TiffFile, imread


def load_videos(
    vpath: str,
    pattern=r"msCam[0-9]+\.avi$",
    dtype: Union[str, type] = np.float64,
    downsample: Optional[dict] = None,
    downsample_strategy="subset",
    post_process: Optional[Callable] = None,
) -> xr.DataArray:
    """
    Load multiple videos in a folder and return a `xr.DataArray`.

    Load videos from the folder specified in `vpath` and according to the regex
    `pattern`, then concatenate them together and return a `xr.DataArray`
    representation of the concatenated videos. The videos are sorted by
    filenames with `natsort` before concatenation. Optionally the data can be
    downsampled, and the user can pass in a custom callable to post-process the
    result.

    Parameters
    ----------
    vpath : str
        the path containing the videos to load
    pattern : regexp, optional
        the regexp matching the filenames of the videso, by default
        r"msCam[0-9]+\.avi$", which can be interpreted as filenames starting
        with "msCam" followed by at least a number, and then followed by ".avi"
    dtype : Union[str, type], optional
        datatype of the resulting DataArray, by default np.float64
    downsample : dict, optional
        a dictionary mapping dimension names to an integer downsampling factor,
        the dimension names should be one of "height", "width" or "frame", by
        default None
    downsample_strategy : str, optional
        how the downsampling should be done, only used if `downsample` is not
        `None`, either "subset" where data points are taken at an interval
        specified in `downsample`, or "mean" where mean will be taken over data
        within each interval, by default "subset"
    post_process : Callable, optional
        an user-supplied custom function to post-process the resulting array,
        four arguments will be passed to the function: the resulting DataArray
        `varr`, the input path `vpath`, the list of matched video filenames
        `vlist`, and the list of DataArray before concatenation `varr_list`, the
        function should output another valide DataArray, in other words, the
        function should have signature `f(varr: xr.DataArray, vpath: str, vlist:
        List[str], varr_list: List[xr.DataArray]) -> xr.DataArray`, by default
        None

    Returns
    -------
    varr : xr.DataArray
        the resulting array representation of the input movie, should have
        dimensions ("frame", "height", "width")

    Raises
    ------
    FileNotFoundError
        if no files under `vpath` match the pattern `pattern`
    ValueError
        if the matched files does not have extension ".avi", ".mkv" or ".tif"
    NotImplementedError
        if `downsample_strategy` is not "subset" or "mean"
    """
    vpath = os.path.normpath(vpath)
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
        varr = post_process(varr, vpath, vlist, varr_list)
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^load_avi_ffmpeg"])
    with da.config.set(array_optimize=arr_opt):
        varr = da.optimize(varr)[0]
    return varr


def load_tif_lazy(fname: str) -> darr.array:
    """
    Lazy load a tif stack of images.

    Parameters
    ----------
    fname : str
        the filename of the tif stack to load

    Returns
    -------
    arr : darr.array
        resulting dask array representation of the tif stack
    """
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


def load_tif_perframe(fname: str, fid: int) -> np.ndarray:
    """
    Load a single image from a tif stack.

    Parameters
    ----------
    fname : str
        the filename of the tif stack
    fid : int
        the index of the image to load

    Returns
    -------
    np.ndarray
        array representation of the image
    """
    return imread(fname, key=fid)


def load_avi_lazy_framewise(fname: str) -> darr.array:
    """
    Lazy load an avi video frame by frame with seeking using `opencv`.

    .. deprecated:: 1.0.0
        `load_avi_lazy_framewise` will be removed in the future in favor of the
        more efficient `load_avi_lazy`.

    Parameters
    ----------
    fname : str
        the filename of the video to load

    Returns
    -------
    arr : darr.array
        the array representation of the video
    """
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


def load_avi_lazy(fname: str) -> darr.array:
    """
    Lazy load an avi video.

    This function construct a single delayed task for loading the video as a
    whole.

    Parameters
    ----------
    fname : str
        the filename of the video to load

    Returns
    -------
    arr : darr.array
        the array representation of the video
    """
    probe = ffmpeg.probe(fname)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w = int(video_info["width"])
    h = int(video_info["height"])
    f = int(video_info["nb_frames"])
    return da.array.from_delayed(
        da.delayed(load_avi_ffmpeg)(fname, h, w, f), dtype=np.uint8, shape=(f, h, w)
    )


def load_avi_ffmpeg(fname: str, h: int, w: int, f: int) -> np.ndarray:
    """
    Load an avi video using `ffmpeg`.

    This function directly invoke `ffmpeg` using the `python-ffmpeg` wrapper and
    retrieve the data from buffer.

    Parameters
    ----------
    fname : str
        the filename of the video to load
    h : int
        the height of the video
    w : int
        the width of the video
    f : int
        the number of frames in the video

    Returns
    -------
    arr : np.ndarray
        the resulting array, has shape (`f`, `h`, `w`)
    """
    out_bytes, err = (
        ffmpeg.input(fname)
        .video.output("pipe:", format="rawvideo", pix_fmt="gray")
        .run(capture_stdout=True)
    )
    return np.frombuffer(out_bytes, np.uint8).reshape(f, h, w)


def load_avi_perframe(fname: str, fid: int) -> np.ndarray:
    """
    Load a single frame in a video with seeking using `opencv`

    .. deprecated:: 1.0.0
        `load_avi_lazy_framewise` will be removed in the future in favor of the
        more efficient `load_avi_lazy`.

    Parameters
    ----------
    fname : str
        the filename to the video
    fid : int
        the index of frame to load

    Returns
    -------
    arr : np.ndarray
        the resulting array
    """
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


def open_minian(
    dpath: str, post_process: Optional[Callable] = None, return_dict=False
) -> Union[dict, xr.Dataset]:
    """
    Load an existing minian dataset.

    This function will iterate through all the directories under input `dpath`
    and load them as `xr.DataArray` with `zarr` backend, so it is important that
    the user make sure every directory under `dpath` can be load this way. The
    loaded arrays will be combined as either a `xr.Dataset` or a `dict`.
    Optionally a user-supplied custom function can be used to post process the
    resulting `xr.Dataset`.

    Parameters
    ----------
    dpath : str
        the path to the minian dataset that should be loaded
    post_process : Callable, optional
        user-supplied function to post process the dataset, only used if
        `return_dict` is `False`, two arguments will be passed to the function:
        the resulting dataset `ds` and the data path `dpath`, in other words the
        function should have signature `f(ds: xr.Dataset, dpath: str) ->
        xr.Dataset`, by default None
    return_dict : bool, optional
        whether to combine the DataArray as dictionary, the `.name` attribute
        will be used as key, otherwise the DataArray will be combined using
        `xr.merge(..., compat="no_conflicts")`, which will implicitly align the
        DataArray over all dimensions, so it is important to make sure the
        coordinates are compatible and will not result in creation of large
        NaN-padded results, by default False

    Returns
    -------
    Union[dict, xr.Dataset]
        the resulting dataset, if `return_dict` is `True` it will be a `dict`,
        otherwise a `xr.Dataset`

    See Also
    -------
    xarray.open_zarr : for how each directory will be loaded as `xr.DataArray`
    xarray.merge : for how the `xr.DataArray` will be merged as `xr.Dataset`
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
    dpath: str,
    index_dims: List[str],
    result_format="xarray",
    pattern=r"minian$",
    sub_dirs: List[str] = [],
    exclude=True,
    **kwargs,
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Open multiple minian datasets across multiple directories.

    This function recursively walks through directories under `dpath` and try to
    load minian datasets from all directories matching `pattern`. It will then
    combine them based on `index_dims` into either a `xr.Dataset` object or a
    `pd.DataFrame`. Optionally a subset of paths can be specified, so that
    they can either be excluded or white-listed.

    Parameters
    ----------
    dpath : str
        the root folder containing all datasets to be loaded
    index_dims : List[str]
        list of dimensions that can be used to index and merge multiple
        datasets, all loaded datasets should have unique coordinates in the
        listed dimensions
    result_format : str, optional
        if "xarray", the result will be merged together recursively along each
        dimensions listed in `index_dims`, users should make sure the
        coordinates are compatible and the merging will not cause generation of
        large NaN-padded results, if "pandas", then a `pd.DataFrame` is
        returned, with columns corresponding to `index_dims` uniquely identify
        each dataset, and an additional column named "minian" of object dtype
        pointing to the loaded minian dataset objects, by default "xarray"
    pattern : regexp, optional
        pattern of minian dataset directory names, by default r"minian$"
    sub_dirs : List[str], optional
        a list of sub-directories under `dpath`, useful if only a subset of
        datasets under `dpath` should be recursively loaded, by default []
    exclude : bool, optional
        whether to exclude directories listed under `sub_dirs`, if `True`, then
        any minian datasets under those specified in `sub_dirs` will be ignored,
        if `False`, then **only** the datasets under those specified in
        `sub_dirs` will be loaded (they still have to be under `dpath` though),
        by default True

    Returns
    -------
    ds : Union[xr.Dataset, pd.DataFrame]
        the resulting combined datasets, if `result_format` is "xarray", then a
        `xr.Dataset` will be returned, otherwise a `pd.DataFrame` will be
        returned

    Raises
    ------
    NotImplementedError
        if `result_format` is not "xarray" or "pandas"

    See Also
    -------
    open_minian
    """
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
    var: xr.DataArray,
    dpath: str,
    meta_dict: Optional[dict] = None,
    overwrite=False,
    chunks: Optional[dict] = None,
    compute=True,
    mem_limit="500MB",
) -> xr.DataArray:
    """
    Save a `xr.DataArray` with `zarr` storage backend following minian
    conventions.

    This function will store arbitrary `xr.DataArray` into `dpath` with `zarr`
    backend. A separate folder will be created under `dpath`, with folder name
    `var.name + ".zarr"`. Optionally metadata can be retrieved from directory
    hierarchy and added as coordinates of the `xr.DataArray`. In addition, an
    on-disk rechunking of the result can be performed using `rechunker` if
    `chunks` are given.

    Parameters
    ----------
    var : xr.DataArray
        the array to be saved
    dpath : str
        the path to the minian dataset directory
    meta_dict : dict, optional
        how metadata should be retrieved from directory hierarchy, the keys
        should be negative integers representing directory level relative to
        `dpath` (so `-1` means the immediate parent directory of `dpath`), and
        values should be the name of dimensions represented by the corresponding
        level of directory, the actual coordinate value of the dimensions will
        be the directory name of corresponding level, by default None
    overwrite : bool, optional
        whether to overwrite the result on disk, by default False
    chunks : dict, optional
        a dictionary specifying the desired chunk size, the chunk size should be
        specified using :doc:`dask:array-chunks` convention, except the "auto"
        specifiication is not supported, the rechunking operation will be
        carried out with on-disk algorithms using `rechunker`, by default None
    compute : bool, optional
        whether to compute `var` and save it immediately, by default True
    mem_limit : str, optional
        the memory limit for the on-disk rechunking algorithm, only used if
        `chunks` is not `None`, by default "500MB"

    Returns
    -------
    var : xr.DataArray
        the array representation of saving result, if `compute` is `True`, then
        the returned array will only contain delayed task of loading the on-disk
        `zarr` arrays, otherwise all computation leading to the input `var` will
        be preserved in the result

    Examples
    -------
    The following will save the variable `var` to directory
    `/spatial_memory/alpha/learning1/minian/important_array.zarr`, with the
    additional coordinates: {"session": "learning1", "animal": "alpha",
    "experiment": "spatial_memory"}.

    >>> save_minian(
    ...     var.rename("important_array"),
    ...     "/spatial_memory/alpha/learning1/minian",
    ...     {-1: "session", -2: "animal", -3: "experiment"},
    ... )
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
    arr = ds.to_zarr(fp, compute=compute, mode=md)
    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))
        with da.config.set(
            array_optimize=darr.optimization.optimize,
            delayed_optimize=default_delay_optimize,
        ):
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
    if compute:
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


def get_optimal_chk(
    arr,
    dim_grp=[("frame",), ("height", "width")],
    csize=256,
    dtype=None,
):
    """
    Estimates the chunk of video (i.e. video sizes and number of frames) that optimizes computer memory use when the script is run parallel over multiple cores.

    Args:
        arr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        dim_grp (array, optional): provide labels for the dimension of the data. Defaults to None.
        csize (int, optional): target chunk size in MB. Defaults to 256.
        dtype (optional): the expected dtype of data. useful when determining chunksize for array with same shape but different dtype.

    Returns:
        dict: sizes of the chunks that optimize memory usage in parallel computing the key is the dimension, the value is the max chunk size
    """
    if dtype is not None:
        arr = arr.astype(dtype)
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]
    chk_compute = dict()
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: "auto" for d in dg}
        dr_dict = {d: -1 for d in d_rest}
        dg_dict.update(dr_dict)
        with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
            arr_chk = arr.chunk(dg_dict)
        chk = get_chunksize(arr_chk)
        chk_compute.update({d: chk[d] for d in dg})
    with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
        arr_chk = arr.chunk({d: "auto" for d in dims})
    chk_store_da = get_chunksize(arr_chk)
    chk_store = dict()
    for d in dims:
        ncomp = int(arr.sizes[d] / chk_compute[d])
        sz = np.array(factors(ncomp)) * chk_compute[d]
        chk_store[d] = sz[np.argmin(np.abs(sz - chk_store_da[d]))]
    return chk_compute, chk_store_da


def get_chunksize(arr):
    dims = arr.dims
    sz = arr.data.chunksize
    return {d: s for d, s in zip(dims, sz)}


def factors(x):
    return [i for i in range(1, x + 1) if x % i == 0]


ANNOTATIONS = {
    "from-zarr-store": {"resources": {"MEM": 1}},
    "load_avi_ffmpeg": {"resources": {"MEM": 1}},
    "est_motion_chunk": {"resources": {"MEM": 1}},
    "transform_perframe": {"resources": {"MEM": 0.5}},
    "pnr_perseed": {"resources": {"MEM": 0.5}},
    "ks_perseed": {"resources": {"MEM": 0.5}},
    "smooth_corr": {"resources": {"MEM": 1}},
    "vectorize_noise_fft": {"resources": {"MEM": 1}},
    "vectorize_noise_welch": {"resources": {"MEM": 1}},
    "update_spatial_block": {"resources": {"MEM": 1}},
    "tensordot_restricted": {"resources": {"MEM": 1}},
    "update_temporal_block": {"resources": {"MEM": 1}},
    "merge_restricted": {"resources": {"MEM": 1}},
}

FAST_FUNCTIONS = [
    darr.core.getter_inline,
    darr.core.getter,
    _operator.getitem,
    zr.core.Array,
    darr.chunk.astype,
    darr.core.concatenate_axes,
    darr.core._vindex_slice,
    darr.core._vindex_merge,
    darr.core._vindex_transpose,
]


class TaskAnnotation(SchedulerPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.annt_dict = ANNOTATIONS

    def update_graph(self, scheduler, client, tasks, **kwargs):
        parent = cast(SchedulerState, scheduler)
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


def custom_arr_optimize(
    dsk,
    keys,
    fast_funcs=FAST_FUNCTIONS,
    inline_patterns=[],
    rename_dict=None,
    rewrite_dict=None,
    keep_patterns=[],
):
    # inlining lots of array operations ref:
    # https://github.com/dask/dask/issues/6668
    if rename_dict:
        key_renamer = fct.partial(custom_fused_keys_renamer, rename_dict=rename_dict)
    else:
        key_renamer = custom_fused_keys_renamer
    keep_keys = []
    if keep_patterns:
        key_ls = list(dsk.keys())
        for pat in keep_patterns:
            keep_keys.extend(list(filter(lambda k: check_key(k, pat), key_ls)))
    dsk = darr.optimization.optimize(
        dsk,
        keys,
        fuse_keys=keep_keys,
        fast_functions=fast_funcs,
        rename_fused_keys=key_renamer,
    )
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if rewrite_dict:
        dsk_old = dsk.copy()
        for key, val in dsk_old.items():
            key_new = rewrite_key(key, rewrite_dict)
            if key_new != key:
                dsk[key_new] = val
                dsk[key] = key_new
    return dsk


def rewrite_key(key, rwdict):
    typ = type(key)
    if typ is tuple:
        k = key[0]
    elif typ is str:
        k = key
    else:
        raise ValueError("key must be either str or tuple: {}".format(key))
    for pat, repl in rwdict.items():
        k = re.sub(pat, repl, k)
    if typ is tuple:
        ret_key = list(key)
        ret_key[0] = k
        return tuple(ret_key)
    else:
        return k


def custom_fused_keys_renamer(keys, max_fused_key_length=120, rename_dict=None):
    """Create new keys for ``fuse`` tasks.

    The optional parameter `max_fused_key_length` is used to limit the maximum string length for each renamed key.
    If this parameter is set to `None`, there is no limit.
    """
    it = reversed(keys)
    first_key = next(it)
    typ = type(first_key)

    if max_fused_key_length:  # Take into account size of hash suffix
        max_fused_key_length -= 5

    def _enforce_max_key_limit(key_name):
        if max_fused_key_length and len(key_name) > max_fused_key_length:
            name_hash = f"{hash(key_name):x}"[:4]
            key_name = f"{key_name[:max_fused_key_length]}-{name_hash}"
        return key_name

    if typ is str:
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key)
        concatenated_name = "-".join(names)
        return _enforce_max_key_limit(concatenated_name)
    elif typ is tuple and len(first_key) > 0 and isinstance(first_key[0], str):
        first_name = split_key(first_key, rename_dict=rename_dict)
        names = {split_key(k, rename_dict=rename_dict) for k in it}
        names.discard(first_name)
        names = sorted(names)
        names.append(first_key[0])
        concatenated_name = "-".join(names)
        return (_enforce_max_key_limit(concatenated_name),) + first_key[1:]


def split_key(key, rename_dict=None):
    if type(key) is tuple:
        key = key[0]
    kls = key.split("-")
    if rename_dict:
        kls = list(map(lambda k: rename_dict.get(k, k), kls))
    kls_ft = list(filter(lambda k: k in ANNOTATIONS.keys(), kls))
    if kls_ft:
        return "-".join(kls_ft)
    else:
        return kls[0]


def check_key(key, pat):
    try:
        return bool(re.search(pat, key))
    except TypeError:
        return bool(re.search(pat, key[0]))


def check_pat(key, pat_ls):
    for pat in pat_ls:
        if check_key(key, pat):
            return True
    return False


def inline_pattern(dsk, pat_ls, inline_constants):
    keys = [k for k in dsk.keys() if check_pat(k, pat_ls)]
    if keys:
        dsk = inline(dsk, keys, inline_constants=inline_constants)
        for k in keys:
            del dsk[k]
        if inline_constants:
            dsk, dep = cull(dsk, set(list(flatten(keys))))
    return dsk


def custom_delay_optimize(dsk, keys, fast_functions=[], inline_patterns=[], **kwargs):
    dsk, _ = fuse(ensure_dict(dsk), rename_keys=custom_fused_keys_renamer)
    if inline_patterns:
        dsk = inline_pattern(dsk, inline_patterns, inline_constants=False)
    if fast_functions:
        dsk = inline_functions(
            dsk,
            [],
            fast_functions=fast_functions,
        )
    return dsk


def unique_keys(keys):
    new_keys = []
    for k in keys:
        if isinstance(k, tuple):
            new_keys.append("chunked-" + k[0])
        elif isinstance(k, str):
            new_keys.append(k)
    return np.unique(new_keys)


def get_keys_pat(pat, keys, return_all=False):
    keys_filt = list(filter(lambda k: check_key(k, pat), list(keys)))
    if return_all:
        return keys_filt
    else:
        return keys_filt[0]


def optimize_chunk(arr, chk):
    fast_funcs = FAST_FUNCTIONS + [darr.core.concatenate3]
    arr_chk = arr.chunk(chk)
    arr_opt = fct.partial(
        custom_arr_optimize,
        fast_funcs=fast_funcs,
        rewrite_dict={"rechunk-merge": "merge_restricted"},
    )
    with da.config.set(array_optimize=arr_opt):
        arr_chk.data = da.optimize(arr_chk.data)[0]
    return arr_chk


def local_extreme(fm, k, etype="max", diff=0):
    fm_max = cv2.dilate(fm, k)
    fm_min = cv2.erode(fm, k)
    fm_diff = ((fm_max - fm_min) > diff).astype(np.uint8)
    if etype == "max":
        fm_ext = (fm == fm_max).astype(np.uint8)
    elif etype == "min":
        fm_ext = (fm == fm_min).astype(np.uint8)
    else:
        raise ValueError("Don't understand {}".format(etype))
    return cv2.bitwise_and(fm_ext, fm_diff).astype(np.uint8)
