import glob
import os
import re
import gc
import time
import matplotlib
import pickle as pkl
import skvideo.io as skv
import skimage.io as ski
import xarray as xr
import numpy as np
import functools as fct
import holoviews as hv
import dask as da
import dask.array.image as daim
import dask.array as darr
import pandas as pd
import subprocess
import warnings
import cv2
import papermill as pm
import ast
import psutil
from pathlib import Path
from dask.diagnostics import ProgressBar
from copy import deepcopy
from scipy import ndimage as ndi
from scipy.io import loadmat
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from collections import Iterable
from tifffile import imsave, imread, TiffFile
from pandas import Timestamp
from IPython.core.debugger import set_trace
from os.path import isdir, abspath
from os import listdir
from os.path import join as pjoin
from itertools import compress

try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as culin

except:
    print("cannot use cuda accelerate")


# def load_params(param):
#     try:
#         param = ast.literal_eval(param)
#     except (ValueError, SyntaxError):
#         pass
#     try:
#         if re.search(r'^slice\([0-9]+, *[0-9]+ *,*[0-9]*\)$', param):
#             param = eval(param)
#     except TypeError:
#         pass
#     if type(param) is dict:
#         param = {k: load_params(v) for k, v in param.items()}
#     return param


def load_params(param):
    try:
        param = eval(param)
    except:
        pass
    if type(param) is dict:
        param = {k: load_params(v) for k, v in param.items()}
    return param


def load_videos(vpath,
                pattern='msCam[0-9]+\.avi$',
                dtype=np.float64,
                in_memory=False,
                downsample=None,
                downsample_strategy='subset',
                post_process=None):
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
    vlist = natsorted([
        vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)
    ])
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath))
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    file_extension = os.path.splitext(vlist[0])[1]
    if file_extension in ('.avi', '.mkv'):
        movie_load_func = load_avi_lazy
    elif file_extension == '.tif':
        movie_load_func = load_tif_lazy
    else:
        raise ValueError('Extension not supported.')

    varr_list = [movie_load_func(v) for v in vlist]
    varr = darr.concatenate(varr_list, axis=0)
    varr = xr.DataArray(
        varr, dims=['frame', 'height', 'width'],
        coords=dict(
            frame=np.arange(varr.shape[0]),
            height=np.arange(varr.shape[1]),
            width=np.arange(varr.shape[2])))
    if dtype:
        varr = varr.astype(dtype)
    if downsample:
        bin_eg = {d: np.arange(0, varr.sizes[d], w)
                  for d, w in downsample.items()}
        if downsample_strategy == 'mean':
            varr = (varr.coarsen(**downsample, boundary='trim')
                    .mean().assign_coords(**bin_eg))
        elif downsample_strategy == 'subset':
            varr = varr.sel(**bin_eg)
        else:
            warnings.warn(
                "unrecognized downsampling strategy", RuntimeWarning)
    varr = varr.rename('fluorescence')
    if post_process:
        varr = post_process(varr, vpath, ssname, vlist, varr_list)
    return varr

def load_tif_lazy(fname):
    data = TiffFile(fname)
    f = len(data.pages)

    fmread = da.delayed(load_tif_perframe)
    flist = [fmread(fname, i) for i in range(f)]

    sample = flist[0].compute()
    arr = [da.array.from_delayed(
        fm, dtype=sample.dtype, shape=sample.shape) for fm in flist]
    return da.array.stack(arr, axis=0)

def load_tif_perframe(fname, fid):
    return imread(fname, key=fid)


def load_avi_lazy(fname):
    cap = cv2.VideoCapture(fname)
    f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fmread = da.delayed(load_avi_perframe)
    flist = [fmread(fname, i) for i in range(f)]
    sample = flist[0].compute()
    arr = [da.array.from_delayed(
        fm, dtype=sample.dtype, shape=sample.shape) for fm in flist]
    return da.array.stack(arr, axis=0)


def load_avi_perframe(fname, fid):
    cap = cv2.VideoCapture(fname)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, fm = cap.read()
    if ret:
        return np.flip(
            cv2.cvtColor(fm, cv2.COLOR_RGB2GRAY), axis=0)
    else:
        print("frame read failed for frame {}".format(fid))
        return np.zeros((h, w))


def load_avi_lazy_pims(fname):
    vid = pims.open(fname)
    f = len(vid)
    def _read_fm(i):
        fm = vid[i]
        r, g, b = fm[:, :, 0], fm[:, :, 1], fm[:, :, 2]
        return np.asarray(0.2125 * r + 0.7154 * g + 0.0721 * b)
    _read_fm_dl = da.delayed(_read_fm)
    flist = [_read_fm_dl(i) for i in range(f)]
    sample = flist[0].compute()
    arr = [da.array.from_delayed(
        fm, dtype=sample.dtype, shape=sample.shape) for fm in flist]
    return da.array.stack(arr, axis=0)


    
def handle_crash(varr, vpath, ssname, vlist, varr_list, frame_dict):
    seg1_list = list(filter(lambda v: re.search('seg1', v), vlist))
    seg2_list = list(filter(lambda v: re.search('seg2', v), vlist))
    if seg1_list and seg2_list:
        tframe = frame_dict[ssname]
        varr1 = darr.concatenate(
            list(compress(varr_list, seg1_list)),
            axis=0)
        varr2 = darr.concatenate(
            list(compress(varr_list, seg2_list)),
            axis=0)
        fm1, fm2 = varr1.shape[0], varr2.shape[0]
        fm_crds = varr.coords['frame']
        fm_crds1 = fm_crds.sel(frame=slice(None, fm1 - 1)).values
        fm_crds2 = fm_crds.sel(frame=slice(fm1, None)).values
        fm_crds2 = fm_crds2 + (tframe - fm_crds2.max())
        fm_crds_new = np.concatenate([fm_crds1, fm_crds2], axis=0)
        return varr.assign_coords(frame=fm_crds_new)
    else:
        return varr


def video_to_tiffs(ipath, opath, iptn='msCam[0-9]+\.avi$', optn='msCam-%05d.tiff'):
    flist = natsorted([os.path.join(ipath, v) for v in os.listdir(ipath) if re.search(iptn, v)])
    istr = "|".join(flist)
    ostr = os.path.join(opath, optn)
    cmd = 'ffmpeg -i "concat:{}" -pix_fmt rgba -compression_algo raw {}'.format(istr, ostr)
    try:
        os.makedirs(opath)
    except OSError:
        if not os.path.isdir(path):
            raise
    print("output directory: {}".format(opath))
    subprocess.check_call(cmd, shell=True)


def load_images(path, dtype=np.float64):
    # imread = fct.partial(ski.imread, as_gray=True)
    imread = fct.partial(imread_cv, dtype=dtype)
    varr = daim.imread(path, imread)
    varr = xr.DataArray(varr, dims=['frame', 'height', 'width'])
    for dim, length in varr.sizes.items():
        varr = varr.assign_coords(**{dim: np.arange(length)})
    return varr


def imread_cv(im, dtype=np.float64):
    return (cv2.imread(im, flags=cv2.IMREAD_GRAYSCALE)
            .astype(dtype))


def create_fig(varlist, nrows, ncols, **kwargs):
    if not isinstance(varlist, list):
        varlist = [varlist]
    if not (nrows or ncols):
        nrows = 1
        ncols = len(varlist)
    elif nrows and not ncols:
        ncols = np.ceil(np.float(len(varlist)) / nrows).astype(int)
    elif ncols and not nrows:
        nrows = np.ceil(np.float(len(varlist)) / ncols).astype(int)
    size = kwargs.pop('size', 5)
    aspect = kwargs.pop('aspect',
                        varlist[0].sizes['width'] / varlist[0].sizes['height'])
    figsize = kwargs.pop('figsize', (aspect * size * ncols, size * nrows))
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if not isinstance(ax, Iterable):
        ax = np.array([ax])
    return fig, ax, varlist, kwargs


def animate_video(varlist, nrows=None, ncols=None, framerate=30, **kwargs):
    fig, ax, varlist, kwargs = create_fig(varlist, nrows, ncols, **kwargs)
    frms = np.min([var.sizes['frame'] for var in varlist])
    f_update = fct.partial(
        multi_im,
        varlist,
        ax=ax,
        add_colorbar=False,
        animated=True,
        tight=False,
        **kwargs)
    f_init = fct.partial(
        multi_im, varlist, subidx={'frame': 0}, ax=ax, **kwargs)
    anm = anim.FuncAnimation(
        fig,
        func=lambda f: f_update(subidx={'frame': f}),
        init_func=f_init,
        frames=frms,
        interval=1000.0 / framerate)
    return fig, anm


def multi_im(varlist,
             subidx=None,
             ax=None,
             nrows=None,
             ncols=None,
             animated=False,
             tight=True,
             **kwargs):
    if ax is None:
        fig, ax, varlist, kwargs = create_fig(varlist, nrows, ncols, **kwargs)
    for ivar, cur_var in enumerate(varlist):
        if subidx:
            va = cur_var.loc[subidx]
        else:
            va = cur_var
        if animated:
            ax[ivar].findobj(matplotlib.collections.QuadMesh)[0].set_array(
                np.ravel(va))
            ax[ivar].set_title("frame = {}".format(int(va.coords['frame'])))
        else:
            ax[ivar].clear()
            va.plot(ax=ax[ivar], **kwargs)
    if tight:
        ax[0].get_figure().tight_layout()
    return ax


def plot_fluorescence(varlist, ax=None, nrows=None, ncols=None, **kwargs):
    if ax is None:
        fig, ax, varlist, kwargs = create_fig(varlist, nrows, ncols, **kwargs)
    for ivar, cur_var in enumerate(varlist):
        cur_mean = cur_var.mean(dim='height').mean(dim='width')
        cur_max = cur_var.max(dim='height').max(dim='width')
        cur_min = cur_var.min(dim='height').min(dim='width')
        ax[ivar].plot(cur_mean.indexes['frame'], cur_mean)
        ax[ivar].fill_between(
            cur_mean.indexes['frame'], cur_min, cur_max, alpha=0.2)
        ax[ivar].set_xlabel('frame')
        ax[ivar].set_ylabel('fluorescence')
        ax[ivar].set_title(cur_var.name)
    return ax


def save_video(movpath, fname_mov_orig, fname_mov_rig, fname_AC, fname_ACbf,
               dsratio):
    """

    Parameters
    ----------
    movpath :

    fname_mov_orig :

    fname_mov_rig :

    fname_AC :

    fname_ACbf :

    dsratio :


    Returns
    -------


    """
    mov_orig = np.load(fname_mov_orig, mmap_mode='r')
    mov_rig = np.load(fname_mov_rig, mmap_mode='r')
    mov_ac = np.load(fname_AC, mmap_mode='r')
    mov_acbf = np.load(fname_ACbf, mmap_mode='r')
    vw = skv.FFmpegWriter(
        movpath, inputdict={'-framerate': '30'}, outputdict={'-r': '30'})
    for fidx in range(0, mov_orig.shape[0], dsratio):
        print("writing frame: " + str(fidx))
        fm_orig = mov_orig[fidx, :, :] * 255
        fm_rig = mov_rig[fidx, :, :] * 255
        fm_acbf = mov_acbf[fidx, :, :] * 255
        fm_ac = mov_ac[fidx, :, :] * 255
        fm = np.concatenate(
            [
                np.concatenate([fm_orig, fm_rig], axis=1),
                np.concatenate([fm_acbf, fm_ac], axis=1)
            ],
            axis=0)
        vw.writeFrame(fm)
    vw.close()


def save_mp4(filename, dat):
    """

    Parameters
    ----------
    filename :

    dat :


    Returns
    -------


    """
    vw = sio.FFmpegWriter(
        filename,
        inputdict={'-framerate': '30'},
        outputdict={
            '-r': '30',
            '-vcodec': 'rawvideo'
        })
    for fid, f in enumerate(dat):
        print("writing frame: {}".format(fid), end='\r')
        vw.writeFrame(f)
    vw.close()


def mov_to_uint8(mov):
    """

    Parameters
    ----------
    mov :


    Returns
    -------



    """
    return np.uint8((mov - np.min(mov)) / (np.max(mov) - np.min(mov)) * 255)


def mov_to_float32(mov):
    """

    Parameters
    ----------
    mov :


    Returns
    -------


    """
    return np.float32((mov - np.min(mov)) / (np.max(mov) - np.min(mov)))


def varr_to_uint8(varr):
    varr_max = varr.max()
    varr_min = varr.min()
    return ((varr - varr_min) / (varr_max - varr_min) * 255).astype(
        np.uint8, copy=False)


def varr_to_float32(varr):
    varr = varr.astype(np.float32, copy=False)
    varr_max = varr.max()
    varr_min = varr.min()
    varr, varr_min_bd = xr.broadcast(varr, varr_min)
    varr_norm = varr - varr_min_bd
    del varr_min_bd
    gc.collect()
    varr_norm, varr_denom = xr.broadcast(varr_norm, (varr_max - varr_min))
    varr_norm = varr_norm / varr_denom
    del varr_denom
    return varr_norm


def scale_varr(varr, scale=(0, 1), inplace=False, pre_compute=False):
    varr_max = varr.max()
    varr_min = varr.min()
    if pre_compute:
        print("pre-computing min and max")
        with ProgressBar():
            varr_max = varr_max.compute()
            varr_min = varr_min.compute()
    if inplace:
        varr_norm = varr
        varr_norm -= varr_min
        varr_norm *= 1 / (varr_max - varr_min)
        varr_norm *= (scale[1] - scale[0])
        varr_norm += scale[0]
    else:
        varr_norm = ((varr - varr_min) * (scale[1] - scale[0])
                     / (varr_max - varr_min)) + scale[0]
    return varr_norm


def scale_varr_da(varr, scale=(0, 1)):
    return ((varr - darr.nanmin(varr)) * (scale[1] - scale[0])
           / (darr.nanmax(varr) - darr.nanmin(varr))) + scale[0]


def normalize(a, scale=(0, 1), copy=False):
    if copy:
        a_norm = a.copy()
    else:
        a_norm = a
    a_max = np.nanmax(a_norm)
    a_min = np.nanmin(a_norm)
    a_norm -= a_min
    a_norm *= 1 / (a_max - a_min)
    a_norm *= (scale[1] - scale[0])
    a_norm += scale[0]
    return a_norm


def varray_to_tif(filename, varr):
    imsave(filename, varr.transpose('frame', 'height', 'width'))


def tif_to_varray(filename):
    arr = imread(filename)
    f = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    varr = xr.DataArray(
        arr,
        coords=dict(frame=range(f), height=range(h), width=range(w)),
        dims=['frame', 'height', 'width'])
    varr.to_netcdf(os.path.dirname(filename) + os.sep + 'varr_mc_int.nc')
    return varr


def resave_varr(path, pattern='^varr_mc_int.tif$'):
    path = os.path.normpath(path)
    tiflist = []
    for dirpath, dirnames, fnames in os.walk(path):
        tifnames = filter(lambda fn: re.search(pattern, fn), fnames)
        tif_paths = [os.path.join(dirpath, tif) for tif in tifnames]
        tiflist += tif_paths
    for itif, tif_path in enumerate(tiflist):
        print("processing {:2d} of {:2d}".format(itif, len(tiflist)), end='\r')
        cur_var = tif_to_varray(tif_path)
        if not cur_var.sizes['height'] == 480 or not cur_var.sizes['width'] == 752:
            print("file {} has modified size: {}".format(
                tif_path, cur_var.sizes))


def plot_varr(varr):
    dvarr = hv.Dataset(varr, kdims=['width', 'height', 'frame'])
    layout = dvarr.to(hv.Image, ['width', 'height'])
    return layout


def save_cnmf(cnmf,
              dpath,
              save_pkl=True,
              from_pkl=False,
              unit_mask=None,
              meta_dict=None,
              order='C'):
    dpath = os.path.normpath(dpath)
    if from_pkl:
        with open(dpath + os.sep + 'cnm.pkl', 'rb') as f:
            cnmf = pkl.load(f)
    else:
        cnmf.dview = None
    if save_pkl:
        with open(dpath + os.sep + 'cnm.pkl', 'wb') as f:
            pkl.dump(cnmf, f)
    varr = xr.open_dataset(dpath + os.sep + 'varr_mc_int.nc')['varr_mc_int']
    f = varr.coords['frame']
    h = varr.coords['height']
    w = varr.coords['width']
    dims = cnmf.dims
    A = xr.DataArray(
        cnmf.A.toarray().reshape(dims + (-1, ), order=order),
        coords={
            'height': h,
            'width': w,
            'unit_id': range(cnmf.A.shape[-1])
        },
        dims=['height', 'width', 'unit_id'],
        name='A')
    C = xr.DataArray(
        cnmf.C,
        coords={
            'unit_id': range(cnmf.C.shape[0]),
            'frame': f
        },
        dims=['unit_id', 'frame'],
        name='C')
    S = xr.DataArray(
        cnmf.S,
        coords={
            'unit_id': range(cnmf.S.shape[0]),
            'frame': f
        },
        dims=['unit_id', 'frame'],
        name='S')
    YrA = xr.DataArray(
        cnmf.YrA,
        coords={
            'unit_id': range(cnmf.S.shape[0]),
            'frame': f
        },
        dims=['unit_id', 'frame'],
        name='YrA')
    b = xr.DataArray(
        cnmf.b.reshape(dims + (-1, ), order=order),
        coords={
            'height': h,
            'width': w,
            'background_id': range(cnmf.b.shape[-1])
        },
        dims=['height', 'width', 'background_id'],
        name='b')
    f = xr.DataArray(
        cnmf.f,
        coords={
            'background_id': range(cnmf.f.shape[0]),
            'frame': f
        },
        dims=['background_id', 'frame'],
        name='f')
    ds = xr.merge([A, C, S, YrA, b, f])
    if from_pkl:
        ds.to_netcdf(dpath + os.sep + 'cnm.nc', mode='a')
    else:
        if unit_mask is None:
            unit_mask = np.arange(ds.sizes['unit_id'])
        if meta_dict is not None:
            pathlist = os.path.normpath(dpath).split(os.sep)
            ds = ds.assign_coords(**dict([(
                cdname,
                pathlist[cdval]) for cdname, cdval in meta_dict.items()]))
        ds = ds.assign_attrs({
            'unit_mask': unit_mask,
            'file_path': dpath + os.sep + "cnm.nc"
        })
        ds.to_netcdf(dpath + os.sep + "cnm.nc")
    return ds


def save_varr(varr, dpath, name='varr_mc_int', meta_dict=None):
    dpath = os.path.normpath(dpath)
    ds = varr.to_dataset(name=name)
    if meta_dict is not None:
        pathlist = os.path.normpath(dpath).split(os.sep)
        ds = ds.assign_coords(**dict([(cdname, pathlist[cdval])
                                      for cdname, cdval in meta_dict.items()]))
    ds = ds.assign_attrs({'file_path': dpath + os.sep + name + '.nc'})
    ds.to_netcdf(dpath + os.sep + name + '.nc')
    return ds


def save_variable(var, fpath, fname, meta_dict=None):
    fpath = os.path.normpath(fpath)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.normpath(fpath).split(os.sep)
        ds = ds.assign_coords(**dict([(cdname, pathlist[cdval])
                                      for cdname, cdval in meta_dict.items()]))
    try:
        ds.to_netcdf(os.path.join(fpath, fname + '.nc'), mode='a')
    except FileNotFoundError:
        ds.to_netcdf(os.path.join(fpath, fname + '.nc'), mode='w')
    return ds


def open_minian(dpath, fname='minian', backend='netcdf', chunks=None, post_process=None):
    if backend is 'netcdf':
        fname = fname + '.nc'
        if chunks is 'auto':
            chunks = dict([(d, 'auto') for d in ds.dims])
        mpath = pjoin(dpath, fname)
        with xr.open_dataset(mpath) as ds:
            dims = ds.dims
        chunks = dict([(d, 'auto') for d in dims])
        ds = xr.open_dataset(os.path.join(dpath, fname), chunks=chunks)
        if post_process:
            ds = post_process(ds, mpath)
        return ds
    elif backend is 'zarr':
        mpath = pjoin(dpath, fname)
        dslist = [xr.open_zarr(pjoin(mpath, d)) for d in listdir(mpath) if isdir(pjoin(mpath, d))]
        ds = xr.merge(dslist)
        if chunks is 'auto':
            chunks = dict([(d, 'auto') for d in ds.dims])
        if post_process:
            ds = post_process(ds, mpath)
        return ds.chunk(chunks)
    else:
        raise NotImplementedError("backend {} not supported".format(backend))


def open_minian_mf(dpath, index_dims, result_format='xarray', pattern=r'minian\.[0-9]+$', sub_dirs=[], exclude=True, **kwargs):
    minian_dict = dict()
    for nextdir, dirlist, filelist in os.walk(dpath, topdown=False):
        nextdir = os.path.abspath(nextdir)
        cur_path = Path(nextdir)
        dir_tag = bool(((any([Path(epath) in cur_path.parents for epath in sub_dirs]))
                        or nextdir in sub_dirs))
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
    if result_format is 'xarray':
        return xrconcat_recursive(minian_dict, index_dims)
    elif result_format is 'pandas':
        minian_df = pd.Series(minian_dict).rename('minian')
        minian_df.index.set_names(index_dims, inplace=True)
        return minian_df.to_frame()
    else:
        raise NotImplementedError(
            "format {} not understood".format(result_format))


def save_minian(var, dpath, fname='minian', backend='netcdf', meta_dict=None, overwrite=False):
    dpath = os.path.normpath(dpath)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.abspath(dpath).split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()]))
    if backend is 'netcdf':
        try:
            md = {True: 'w', False: 'a'}[overwrite]
            ds.to_netcdf(os.path.join(dpath, fname + '.nc'), mode=md)
        except FileNotFoundError:
            ds.to_netcdf(os.path.join(dpath, fname + '.nc'), mode=md)
        return ds
    elif backend is 'zarr':
        md = {True: 'w', False: 'w-'}[overwrite]
        fp = os.path.join(dpath, fname, var.name + '.zarr')
        ds.to_zarr(fp, mode=md)
        return xr.open_zarr(fp)[var.name]
    else:
        raise NotImplementedError("backend {} not supported".format(backend))

def delete_variable(fpath, varlist, del_org=False):
    fpath_bak = fpath + ".{}.backup".format(int(time.time()))
    os.rename(fpath, fpath_bak)
    with xr.open_dataset(fpath_bak) as ds:
        new_ds = ds.drop(varlist)
        new_ds.to_netcdf(fpath)
    if del_org:
        os.remove(fpath_bak)
    return "deleted {} in file {}".format(str(varlist), fpath)


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


def update_meta(dpath, pattern=r'^minian\.nc$', meta_dict=None, backend='netcdf'):
    for dirpath, dirnames, fnames in os.walk(dpath):
        if backend == 'netcdf':
            fnames = filter(lambda fn: re.search(pattern, fn), fnames)
        elif backend == 'zarr':
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
            new_ds = new_ds.assign_coords(**dict([(
                cdname,
                pathlist[cdval]) for cdname, cdval in meta_dict.items()]))
            if backend == 'netcdf':
                new_ds.to_netcdf(f_path, mode='a')
            elif backend == 'zarr':
                new_ds.to_zarr(f_path, mode='w')
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


def get_optimal_chk(ref, arr=None, dim_grp=None, ncores='auto', mem_limit='auto'):
    if arr is None:
        arr = ref
    szs = ref.sizes
    if ncores=='auto':
        ncores = psutil.cpu_count()
    if mem_limit=='auto':
        mem_limit = (psutil.virtual_memory().available / (1024 ** 2))
    tempsz = 1000*(3*szs['height'] * szs['width'] + 7 * szs['frame']) * ref.dtype.itemsize / (1024 ** 2)
    csize = min(int(np.floor((mem_limit - tempsz) / ncores / 4)), 1024)
    if csize <= 0:
        warnings.warn(
            "estimated memory limit is smaller than 0. Using 64MiB chunksize instead. "
            "Make sure you have enough memory or manually set mem_limit")
        csize = 64
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]
    opt_chk = dict()
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: 'auto' for d in dg}
        dr_dict = {d: -1 for d in d_rest}
        dg_dict.update(dr_dict)
        with da.config.set({'array.chunk-size': '{}MiB'.format(csize)}):
            arr_chk = arr.chunk(dg_dict)
        re_dict = {d: c for d, c in zip(dims, arr_chk.chunks)}
        re_dict = {d: max(re_dict[d]) for d in dg}
        opt_chk.update(re_dict)
    return opt_chk

# def resave_varr_again(dpath, pattern=r'^varr_mc_int.nc$'):
#     for dirpath, dirnames, fnames in os.walk(dpath):
#         fnames = filter(lambda fn: re.search(pattern, fn), fnames)
#         for fname in fnames:
#             f_path = os.path.join(dirpath, fname)
#             with xr.open_dataset(f_path) as old_ds:
#                 vname = list(old_ds.data_vars.keys())[0]
#                 if vname == 'varr_mc_int':
#                     continue
#                 print("resaving {}".format(f_path))
#                 ds = old_ds.load().copy()
#                 ds = ds.rename({vname: 'varr_mc_int'})
#             ds.to_netcdf(f_path, mode='w')

# def resave_cnmf(dpath, pattern=r'^cnm.nc$'):
#     for dirpath, fdpath, fpath in os.walk(dpath):
#         f_list = filter(lambda fn: re.search(pattern, fn), fpath)
#         for cnm_path in f_list:
#             cnm_path = os.path.join(dirpath, cnm_path)
#             cur_cnm = xr.open_dataset(cnm_path)
#             newds = xr.Dataset()
#             newds.assign_coords(session=cur_cnm.coords['session'])
#             newds.assign_coords(animal=cur_cnm.coords['animal'])
#             newds.assign_coords(session_id=cur_cnm.coords['session_id'])
#             fpath = str(cur_cnm.attrs['file_path'])
#             cur_cnm.close()
#             print("writing to ".format(fpath))
#             newds.to_netcdf(fpath, mode='a')


def save_movies(cnmf, dpath, Y=None, mask=None, Y_only=True, order='C'):
    try:
        cnmd = vars(cnmf)
    except TypeError:
        cnmd = cnmf
    dims = cnmd['dims']
    if not Y_only:
        print("calculating A * C")
        if mask is not None:
            A_dot_C = cnmd['A'].toarray()[:, mask].dot(
                cnmd['C'][mask, :]).astype(np.float32)
        else:
            A_dot_C = cnmd['A'].toarray().dot(cnmd['C']).astype(np.float32)
        print("calculating b * f")
        b_dot_f = cnmd['b'].dot(cnmd['f']).astype(np.float32)
        A_dot_C = xr.DataArray(
            A_dot_C.reshape(dims + (-1, ), order=order),
            coords={
                'height': range(dims[0]),
                'width': range(dims[1]),
                'frame': range(A_dot_C.shape[-1])
            },
            dims=['height', 'width', 'frame'],
            name='A_dot_C')
        b_dot_f = xr.DataArray(
            b_dot_f.reshape(dims + (-1, ), order=order),
            coords={
                'height': range(dims[0]),
                'width': range(dims[1]),
                'frame': range(b_dot_f.shape[-1])
            },
            dims=['height', 'width', 'frame'],
            name='b_dot_f')
    if Y is not None:
        Y = np.moveaxis(Y.astype(np.float32), 0, -1)
        if not isinstance(Y, xr.DataArray):
            Y = xr.DataArray(
                Y,
                coords={
                    'height': range(Y.shape[0]),
                    'width': range(Y.shape[1]),
                    'frame': range(Y.shape[2])
                },
                dims=['height', 'width', 'frame'],
                name='Y')
        if not Y_only:
            print("calculating Yres")
            Yres = Y.copy()
            Yres -= A_dot_C
            Yres -= b_dot_f
            Yres = Yres.rename('Yres')
    else:
        Yres = None
    if not Y_only:
        print("merging")
        ds = xr.merge([Y, A_dot_C, b_dot_f, Yres])
    else:
        ds = Y
    print("writing to disk")
    ds.to_netcdf(dpath + os.sep + "movies.nc")
    return ds


def save_cnmf_from_mat(matpath,
                       dpath,
                       vname="ms",
                       order='C',
                       dims=None,
                       T=None,
                       unit_mask=None,
                       meta_dict=None):
    dpath = os.path.normpath(dpath)
    mat = loadmat(matpath, squeeze_me=True, struct_as_record=False)
    try:
        cnmf = mat[vname]
    except KeyError:
        print("No variable with name {} was found in the .mat file: {}".format(
            vname, matpath))
        return
    if not dims:
        dims = (cnmf.options.d1, cnmf.options.d2)
        dims_coord = (list(range(dims[0])), list(range(dims[1])))
    else:
        dims_coord = (np.linspace(0, dims[0] - 1, cnmf.options.d1),
                      np.linspace(0, dims[1] - 1, cnmf.options.d2))
        dims = (cnmf.options.d1, cnmf.options.d2)
    if not T:
        T = cnmf.C.shape[1]
        T_coord = list(range(T))
    else:
        T_coord = np.linspace(0, T - 1, cnmf.C.shape[1])
        T = cnmf.C.shape[1]
    A = xr.DataArray(
        cnmf.A.reshape(dims + (-1, ), order=order),
        coords={
            'height': dims_coord[0],
            'width': dims_coord[1],
            'unit_id': range(cnmf.A.shape[-1])
        },
        dims=['height', 'width', 'unit_id'],
        name='A')
    C = xr.DataArray(
        cnmf.C,
        coords={
            'unit_id': range(cnmf.C.shape[0]),
            'frame': T_coord
        },
        dims=['unit_id', 'frame'],
        name='C')
    S = xr.DataArray(
        cnmf.S,
        coords={
            'unit_id': range(cnmf.S.shape[0]),
            'frame': T_coord
        },
        dims=['unit_id', 'frame'],
        name='S')
    if cnmf.b.any():
        b = xr.DataArray(
            cnmf.b.reshape(dims + (-1, ), order=order),
            coords={
                'height': dims_coord[0],
                'width': dims_coord[1],
                'background_id': range(cnmf.b.shape[-1])
            },
            dims=['height', 'width', 'background_id'],
            name='b')
    else:
        b = xr.DataArray(
            np.zeros(dims + (1, )),
            coords=dict(
                height=dims_coord[0], width=dims_coord[1], background_id=[0]),
            dims=['height', 'width', 'background_id'],
            name='b')
    if cnmf.f.any():
        f = xr.DataArray(
            cnmf.f,
            coords={
                'background_id': range(cnmf.f.shape[0]),
                'frame': T_coord
            },
            dims=['background_id', 'frame'],
            name='f')
    else:
        f = xr.DataArray(
            np.zeros((1, T)),
            coords=dict(background_id=[0], frame=T_coord),
            dims=['background_id', 'frame'],
            name='f')
    ds = xr.merge([A, C, S, b, f])
    if unit_mask is None:
        unit_mask = np.arange(ds.sizes['unit_id'])
    if meta_dict is not None:
        pathlist = os.path.normpath(dpath).split(os.sep)
        ds = ds.assign_coords(
            **{cdname: pathlist[cdval]
               for cdname, cdval in meta_dict.items()})
    ds = ds.assign_attrs({
        'unit_mask': unit_mask,
        'file_path': dpath + os.sep + "cnm_from_mat.nc"
    })
    ds.to_netcdf(dpath + os.sep + "cnm_from_mat.nc")
    return ds