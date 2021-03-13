import functools as fct
import itertools as itt

import cv2
import dask as da
import dask.array as darr
import numpy as np
import xarray as xr

from .utilities import custom_arr_optimize, xrconcat_recursive


def estimate_shifts(varr, max_sh, dim="frame", npart=None, local=False, temp_nfm=30):
    """
    Estimate the frame shifts

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        max_sh (integer): maximum shift
        dim (str, optional): name of the z-dimension, defaults to "frame".
        npart (integer, optional): [the number of partitions of the divide-and-conquer algorithm]. Defaults to 3.
        local (boolean, optional): [in case where there are multiple local maximum of the cross-correlogram, setting this to `True` will constraint the shift to be the one that’s closest to zero shift. i.e. this assumes the shifts are always small and local regardless of the correlation value]. Defaults to False.

    Returns:
        xarray.DataArray: the estimated shifts
    """
    varr = varr.transpose(..., dim, "height", "width")
    loop_dims = list(set(varr.dims) - set(["height", "width", dim]))
    if npart is None:
        # by default use a npart that result in two layers of recursion
        npart = max(3, int(np.ceil((varr.sizes[dim] / temp_nfm) ** (1 / 2))))
    if loop_dims:
        loop_labs = [varr.coords[d].values for d in loop_dims]
        res_dict = dict()
        for lab in itt.product(*loop_labs):
            va = varr.sel({loop_dims[i]: lab[i] for i in range(len(loop_dims))})
            vmax, sh = est_sh_part(va.data, max_sh, npart, local, temp_nfm)
            sh = xr.DataArray(
                sh,
                dims=[dim, "variable"],
                coords={dim: va.coords[dim].values, "variable": ["height", "width"],},
            )
            res_dict[lab] = sh.assign_coords(**{k: v for k, v in zip(loop_dims, lab)})
        sh = xrconcat_recursive(res_dict, loop_dims)
    else:
        vmax, sh = est_sh_part(varr.data, max_sh, npart, local, temp_nfm)
        sh = xr.DataArray(
            sh,
            dims=[dim, "variable"],
            coords={dim: varr.coords[dim].values, "variable": ["height", "width"],},
        )
    return sh


def est_sh_part(varr, max_sh, npart, local, temp_nfm):
    """
    Estimate the shift per frame

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        max_sh (integer): maximum shift
        npart (integer): [the number of partitions of the divide-and-conquer algorithm].
        local (boolean): [in case where there are multiple local maximum of the cross-correlogram, setting this to `True` will constraint the shift to be the one that’s closest to zero shift. i.e. this assumes the shifts are always small and local regardless of the correlation value].

    Returns:
        xarray.DataArray: the max shift per frame
        xarray.DataArray: the shift per frame
    """
    varr = varr.rechunk((temp_nfm, None, None))
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^est_sh_chunk"])
    tmp_ls = []
    sh_ls = []
    for blk in varr.blocks:
        res = da.delayed(est_sh_chunk)(blk, sh_org=None, max_sh=max_sh, local=local)
        tmp = darr.from_delayed(
            res[0], shape=(blk.shape[1], blk.shape[2]), dtype=blk.dtype
        )
        sh = darr.from_delayed(res[1], shape=(blk.shape[0], 2), dtype=float)
        tmp_ls.append(tmp)
        sh_ls.append(sh)
    with da.config.set(array_optimize=arr_opt):
        temps = da.optimize(darr.stack(tmp_ls, axis=0))[0]
        shifts = da.optimize(darr.concatenate(sh_ls, axis=0))[0]
    while temps.shape[0] > 1:
        tmp_ls = []
        sh_ls = []
        for idx in np.arange(0, temps.numblocks[0], npart):
            tmps = temps.blocks[idx : idx + npart]
            sh_org = shifts.blocks[idx : idx + npart]
            sh_org_ls = [sh_org.blocks[i] for i in range(sh_org.numblocks[0])]
            res = da.delayed(est_sh_chunk)(tmps, sh_org_ls, max_sh=max_sh, local=local)
            tmp = darr.from_delayed(
                res[0], shape=(tmps.shape[1], tmps.shape[2]), dtype=tmps.dtype
            )
            sh_new = darr.from_delayed(res[1], shape=sh_org.shape, dtype=sh_org.dtype)
            tmp_ls.append(tmp)
            sh_ls.append(sh_new)
        temps = darr.stack(tmp_ls, axis=0)
        shifts = darr.concatenate(sh_ls, axis=0)
    return temps, shifts


def est_sh_chunk(varr, sh_org, max_sh, local):
    mid = int(varr.shape[0] / 2)
    shifts = np.zeros((varr.shape[0], 2))
    for i, fm in enumerate(varr):
        if i < mid:
            temp = varr[i + 1]
            slc = slice(0, i + 1)
        elif i > mid:
            temp = varr[i - 1]
            slc = slice(i, None)
        else:
            continue
        sh = match_temp(fm, temp, max_sh, local)
        shifts[slc] = shifts[slc] + sh
    for i, sh in enumerate(shifts):
        varr[i] = shift_perframe(varr[i], sh, 0)
    if sh_org is not None:
        shifts = np.concatenate([shifts[i] + sh for i, sh in enumerate(sh_org)], axis=0)
    return varr.max(axis=0), shifts


def match_temp(src, dst, max_sh, local, subpixel=False):
    """
    Match template.
    For more information on template matching: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

    Args:
        src (array): source frame
        dst (array): template
        max_sh (integer): maximum shift
        local (boolean): [in case where there are multiple local maximum of the cross-correlogram, setting this to `True` will constraint the shift to be the one that’s closest to zero shift. i.e. this assumes the shifts are always small and local regardless of the correlation value].
        subpixel (boolean, optional): [whether to estimate shifts to sub-pixel level using polynomial fitting of cross-correlogram]. Defaults to False.

    Returns:
        [array]: array (x,y) of the shift (match)
    """
    src = np.pad(src, max_sh)
    cor = cv2.matchTemplate(
        src.astype(np.float32), dst.astype(np.float32), cv2.TM_CCOEFF_NORMED
    )
    if not len(np.unique(cor)) > 1:
        return np.array([0, 0])
    cent = np.floor(np.array(cor.shape) / 2)
    if local:
        cor_ma = cv2.dilate(cor, np.ones((3, 3)))
        maxs = np.array(np.nonzero(cor_ma == cor))
        dev = ((maxs - cent[:, np.newaxis]) ** 2).sum(axis=0)
        imax = maxs[:, np.argmin(dev)]
    else:
        imax = np.unravel_index(np.argmax(cor), cor.shape)
    if subpixel:
        x0 = np.arange(max(imax[0] - 5, 0), min(imax[0] + 6, cor.shape[0]))
        x1 = np.arange(max(imax[1] - 5, 0), min(imax[1] + 6, cor.shape[1]))
        y0 = cor[x0, imax[1]]
        y1 = cor[imax[0], x1]
        p0 = np.polyfit(x0, y0, 2)
        p1 = np.polyfit(x1, y1, 2)
        imax = np.array([-0.5 * p0[1] / p0[0], -0.5 * p1[1] / p1[0]])
        # m0 = np.roots(np.polyder(p0)) # for higher order polynomial fit
        # m1 = np.roots(np.polyder(p1))
        # m0 = m0[np.argmin(np.abs(m0 - imax[0]))]
        # m1 = m1[np.argmin(np.abs(m1 - imax[1]))]
        # imax = np.array([m0, m1])
    sh = cent - imax
    return sh


def apply_shifts(varr, shifts, fill=np.nan):
    """
    Apply the shifts to the input frames

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        shifts (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the shifts with dimensions: frame, height and width.

    Returns:
        (xarray.DataArray): xarray.DataArray of the shifted input frames (varr)
    """
    sh_dim = shifts.coords["variable"].values.tolist()
    varr_sh = xr.apply_ufunc(
        shift_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        shifts,
        input_core_dims=[sh_dim, ["variable"]],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask="parallelized",
        kwargs={"fill": fill},
        output_dtypes=[varr.dtype],
    )
    return varr_sh


def shift_perframe(fm, sh, fill=np.nan):
    """
    Determine the shift per frame

    Args:
        fm (array): array with the pixels of the frame
        sh (array): (x,y) shift

    Returns:
        array: frame
    """
    if np.isnan(fm).all():
        return fm
    sh = np.around(sh).astype(int)
    fm = np.roll(fm, sh, axis=np.arange(fm.ndim))
    index = [slice(None) for _ in range(fm.ndim)]
    for ish, s in enumerate(sh):
        index = [slice(None) for _ in range(fm.ndim)]
        if s > 0:
            index[ish] = slice(None, s)
            fm[tuple(index)] = fill
        elif s == 0:
            continue
        elif s < 0:
            index[ish] = slice(s, None)
            fm[tuple(index)] = fill
    return fm
