import os
import re
import warnings
from copy import deepcopy
from os import listdir
from os.path import isdir
from os.path import join as pjoin
from pathlib import Path

import cv2
import dask as da
import dask.array as darr
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from natsort import natsorted
from tifffile import TiffFile, imread


def load_videos(
    vpath,
    pattern=r"msCam[0-9]+\.avi$",
    dtype=np.float64,
    in_memory=False,
    downsample=None,
    downsample_strategy="subset",
    post_process=None,
):
    """Load videos from a folder.

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
        bin_eg = {d: np.arange(0, varr.sizes[d], w) for d, w in downsample.items()}
        if downsample_strategy == "mean":
            varr = (
                varr.coarsen(**downsample, boundary="trim")
                .mean()
                .assign_coords(**bin_eg)
            )
        elif downsample_strategy == "subset":
            varr = varr.sel(**bin_eg)
        else:
            warnings.warn("unrecognized downsampling strategy", RuntimeWarning)
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


def load_avi_lazy(fname):
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


def open_minian(
    dpath, fname="minian", backend="netcdf", chunks=None, post_process=None
):
    if backend == "netcdf":
        fname = fname + ".nc"
        if chunks == "auto":
            chunks = dict([(d, "auto") for d in ds.dims])
        mpath = pjoin(dpath, fname)
        with xr.open_dataset(mpath) as ds:
            dims = ds.dims
        chunks = dict([(d, "auto") for d in dims])
        ds = xr.open_dataset(os.path.join(dpath, fname), chunks=chunks)
        if post_process:
            ds = post_process(ds, mpath)
        return ds
    elif backend == "zarr":
        mpath = pjoin(dpath, fname)
        dslist = [
            xr.open_zarr(pjoin(mpath, d))
            for d in listdir(mpath)
            if isdir(pjoin(mpath, d))
        ]
        ds = xr.merge(dslist)

        if chunks == "auto":
            chunks = dict([(d, "auto") for d in ds.dims])
        if post_process:
            ds = post_process(ds, mpath)
        return ds.chunk(chunks)
    else:
        raise NotImplementedError("backend {} not supported".format(backend))


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
            minian = open_minian(nextdir, fname=fname, **kwargs)
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
    var, dpath, fname="minian", backend="netcdf", meta_dict=None, overwrite=False
):
    dpath = os.path.normpath(dpath)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.abspath(dpath).split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
        )
    if backend == "netcdf":
        try:
            md = {True: "w", False: "a"}[overwrite]
            ds.to_netcdf(os.path.join(dpath, fname + ".nc"), mode=md)
        except FileNotFoundError:
            ds.to_netcdf(os.path.join(dpath, fname + ".nc"), mode=md)
        return ds
    elif backend == "zarr":
        md = {True: "w", False: "w-"}[overwrite]
        fp = os.path.join(dpath, fname, var.name + ".zarr")
        ds.to_zarr(fp, mode=md)
        return xr.open_zarr(fp)[var.name]
    else:
        raise NotImplementedError("backend {} not supported".format(backend))


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
    try:
        dst_chk = get_chk(y)
        comm_dim = set(x.dims).intersection(set(dst_chk.keys()))
        dst_chk = {d: max(dst_chk[d]) for d in comm_dim}
        return x.chunk(dst_chk)
    except TypeError:
        return x.compute()


def get_optimal_chk(ref, arr=None, dim_grp=None, ncores="auto", mem_limit="auto"):
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
