import functools as fct
import itertools as itt
import warnings

import cv2
import dask as da
import dask.array as darr
import numpy as np
import SimpleITK as sitk
import xarray as xr
from skimage.registration import phase_cross_correlation

from .utilities import custom_arr_optimize, xrconcat_recursive


def estimate_motion(varr, dim="frame", npart=3, temp_nfm=None, **kwargs):
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
            vmax, sh = est_motion_part(va.data, npart, temp_nfm, **kwargs)
            if kwargs.get("mesh_size", None):
                sh = xr.DataArray(
                    sh,
                    dims=[dim, "shift_dim", "grid0", "grid1"],
                    coords={
                        dim: va.coords[dim].values,
                        "shift_dim": ["height", "width"],
                    },
                )
            else:
                sh = xr.DataArray(
                    sh,
                    dims=[dim, "shift_dim"],
                    coords={
                        dim: va.coords[dim].values,
                        "shift_dim": ["height", "width"],
                    },
                )
            res_dict[lab] = sh.assign_coords(**{k: v for k, v in zip(loop_dims, lab)})
        sh = xrconcat_recursive(res_dict, loop_dims)
    else:
        vmax, sh = est_motion_part(varr.data, npart, temp_nfm, **kwargs)
        if kwargs.get("mesh_size", None):
            sh = xr.DataArray(
                sh,
                dims=[dim, "shift_dim", "grid0", "grid1"],
                coords={
                    dim: varr.coords[dim].values,
                    "shift_dim": ["height", "width"],
                },
            )
        else:
            sh = xr.DataArray(
                sh,
                dims=[dim, "shift_dim"],
                coords={
                    dim: varr.coords[dim].values,
                    "shift_dim": ["height", "width"],
                },
            )
    return sh


def est_motion_part(varr, npart, temp_nfm, alt_error=5, **kwargs):
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
    if temp_nfm is None:
        temp_nfm = varr.chunksize[0]
    varr = varr.rechunk((temp_nfm, None, None))
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^est_motion_chunk"])
    if kwargs.get("mesh_size", None):
        param = get_bspline_param(varr[0].compute(), kwargs["mesh_size"])
    tmp_ls = []
    sh_ls = []
    for blk in varr.blocks:
        res = da.delayed(est_motion_chunk)(
            blk, None, alt_error=alt_error, npart=npart, **kwargs
        )
        if alt_error:
            tmp = darr.from_delayed(
                res[0], shape=(3, blk.shape[1], blk.shape[2]), dtype=blk.dtype
            )
        else:
            tmp = darr.from_delayed(
                res[0], shape=(blk.shape[1], blk.shape[2]), dtype=blk.dtype
            )
        if kwargs.get("mesh_size", None):
            sh = darr.from_delayed(
                res[1],
                shape=(blk.shape[0], 2, int(param[1]), int(param[0])),
                dtype=float,
            )
        else:
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
            res = da.delayed(est_motion_chunk)(
                tmps, sh_org_ls, alt_error=alt_error, npart=npart, **kwargs
            )
            if alt_error:
                tmp = darr.from_delayed(
                    res[0], shape=(3, tmps.shape[1], tmps.shape[2]), dtype=tmps.dtype
                )
            else:
                tmp = darr.from_delayed(
                    res[0], shape=(tmps.shape[1], tmps.shape[2]), dtype=tmps.dtype
                )
            sh_new = darr.from_delayed(res[1], shape=sh_org.shape, dtype=sh_org.dtype)
            tmp_ls.append(tmp)
            sh_ls.append(sh_new)
        temps = darr.stack(tmp_ls, axis=0)
        shifts = darr.concatenate(sh_ls, axis=0)
    return temps, shifts


def est_motion_chunk(
    varr,
    sh_org,
    npart,
    alt_error,
    aggregation="mean",
    upsample=100,
    max_sh=100,
    circ_thres=None,
    mesh_size=None,
    niter=100,
    bin_thres=None,
):
    if varr.shape[0] == 1:
        if sh_org is not None:
            motions = sh_org
        else:
            if mesh_size is not None:
                # TODO handle non-rigid case
                pass
            else:
                motions = np.array([0, 0])[np.newaxis, :]
        if alt_error:
            tmp = np.stack([varr[0]] * 3)
        else:
            tmp = varr[0]
        return tmp, motions
    while varr.shape[0] > npart:
        part_idx = np.array_split(
            np.arange(varr.shape[0]), np.ceil(varr.shape[0] / npart)
        )
        tmp_ls = []
        sh_ls = []
        for idx in part_idx:
            cur_tmp, cur_motions = est_motion_chunk(
                varr[idx],
                [sh_org[i] for i in idx] if sh_org is not None else None,
                npart=npart,
                alt_error=alt_error,
                aggregation=aggregation,
                upsample=upsample,
                max_sh=max_sh,
                circ_thres=circ_thres,
                mesh_size=mesh_size,
                niter=niter,
                bin_thres=bin_thres,
            )
            tmp_ls.append(cur_tmp)
            sh_ls.append(cur_motions)
        varr = np.stack(tmp_ls, axis=0)
        sh_org = sh_ls
    # varr could have 4 dimensions in which case the second dimension has length
    # 3 representing the first, aggregated and the last frame of a chunk
    mask = np.ones_like(varr, dtype=bool)
    if bin_thres is not None and varr.ndim <= 3:
        for i, fm in enumerate(varr):
            mask[i] = fm > bin_thres
    good_fm = np.ones(varr.shape[0], dtype=bool)
    if circ_thres is not None and varr.ndim <= 3:
        for i, fm in enumerate(varr):
            good_fm[i] = check_temp(fm, max_sh) > circ_thres
    good_idxs = np.where(good_fm)[0].astype(int)
    prop_good = len(good_idxs) / len(good_fm)
    if prop_good < 0.9:
        warnings.warn(
            "only {} of the frames are good."
            "Consider lowering your circularity threshold".format(prop_good)
        )
    # use good frame closest to center as template
    mid = good_idxs[np.abs(good_idxs - varr.shape[0] / 2).argmin()]
    if mesh_size is not None:
        fm0 = varr[0, 0] if varr.ndim > 3 else varr[0]
        param = get_bspline_param(fm0, mesh_size)
        motions = np.zeros((varr.shape[0], 2, int(param[1]), int(param[0])))
    else:
        motions = np.zeros((varr.shape[0], 2))
    for i, fm in enumerate(varr):
        if i < mid:
            if varr.ndim > 3:
                src, dst = varr[i][1], varr[i + 1][1]
                src_ma, dst_ma = mask[i][1], mask[i + 1][1]
                if alt_error:
                    src_alt, dst_alt = varr[i][2], varr[i + 1][0]
                    src_alt_ma, dst_alt_ma = mask[i][2], mask[i + 1][0]
            else:
                # select the next good frame as template
                didx = good_idxs[good_idxs - (i + 1) >= 0][0]
                src, dst = varr[i], varr[didx]
                src_ma, dst_ma = mask[i], mask[didx]
            slc = slice(0, i + 1)
        elif i > mid:
            if varr.ndim > 3:
                src, dst = varr[i][1], varr[i - 1][1]
                src_ma, dst_ma = mask[i][1], mask[i - 1][1]
                if alt_error:
                    src_alt, dst_alt = varr[i][0], varr[i - 1][2]
                    src_alt_ma, dst_alt_ma = mask[i][0], mask[i - 1][2]
            else:
                # select the previous good frame as template
                didx = good_idxs[good_idxs - (i - 1) <= 0][-1]
                src, dst = varr[i], varr[didx]
                src_ma, dst_ma = mask[i], mask[didx]
            slc = slice(i, None)
        else:
            continue
        mo = est_motion_perframe(src, dst, upsample, src_ma, dst_ma, mesh_size, niter)
        if alt_error and varr.ndim > 3:
            mo_alt = est_motion_perframe(
                src_alt, dst_alt, upsample, src_alt_ma, dst_alt_ma, mesh_size, niter
            )
            if ((np.abs(mo - mo_alt) > alt_error).any()) and (
                np.abs(mo).sum() > np.abs(mo_alt).sum()
            ):
                mo = mo_alt
        # only add to the rest if current frame is good
        if good_fm[i]:
            motions[slc] = motions[slc] + mo
        else:
            motions[i] = motions[i] + mo
    # center shifts
    if mesh_size is not None:
        motions -= motions.mean(axis=(0, 2, 3), keepdims=True)
    else:
        motions -= motions.mean(axis=0)
    for i, v in enumerate(varr):
        if i not in good_idxs:
            continue
        if v.ndim > 2:
            for j, fm in enumerate(v):
                varr[i][j] = transform_perframe(fm, motions[i], fill=0)
        else:
            varr[i] = transform_perframe(v, motions[i], fill=0)
    varr = varr[good_idxs]
    if aggregation == "max":
        if varr.ndim > 3:
            tmp = varr.max(axis=(0, 1))
        else:
            tmp = varr.max(axis=0)
    elif aggregation == "mean":
        if varr.ndim > 3:
            tmp = varr.mean(axis=(0, 1))
        else:
            tmp = varr.mean(axis=0)
    else:
        raise ValueError("does not understand aggregation: {}".format(aggregation))
    if alt_error:
        if varr.ndim > 3:
            tmp0 = varr[0][0]
            tmp1 = varr[-1][-1]
        else:
            tmp0 = varr[0]
            tmp1 = varr[1]
        tmp = np.stack([tmp0, tmp, tmp1], axis=0)
    if sh_org is not None:
        motions = np.concatenate(
            [motions[i] + sh for i, sh in enumerate(sh_org)], axis=0
        )
    return tmp, motions


def est_motion_perframe(
    src, dst, upsample, src_ma=None, dst_ma=None, mesh_size=None, niter=100
):
    sh = phase_cross_correlation(
        src,
        dst,
        upsample_factor=upsample,
        return_error=False,
    )
    if mesh_size is None:
        return -sh
    src = sitk.GetImageFromArray(src.astype(np.float32))
    dst = sitk.GetImageFromArray(dst.astype(np.float32))
    reg = sitk.ImageRegistrationMethod()
    sh = sh[::-1]
    trans_init = sitk.TranslationTransform(2, sh)
    reg.SetMovingInitialTransform(trans_init)
    if src_ma is not None:
        reg.SetMetricMovingMask(sitk.GetImageFromArray(src_ma.astype(np.uint8)))
    if dst_ma is not None:
        reg.SetMetricFixedMask(sitk.GetImageFromArray(dst_ma.astype(np.uint8)))
    trans_opt = sitk.BSplineTransformInitializer(
        image1=dst, transformDomainMeshSize=mesh_size
    )
    reg.SetInitialTransform(trans_opt)
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.1, convergenceMinimumValue=1e-5, numberOfIterations=niter
    )
    tx = reg.Execute(dst, src)
    coef = np.stack(
        [sitk.GetArrayFromImage(im) for im in tx.Downcast().GetCoefficientImages()]
    )
    coef = coef + sh.reshape((2, 1, 1))
    return coef


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
    dst = np.pad(dst, max_sh)
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


def check_temp(fm, max_sh):
    fm_pad = np.pad(fm, max_sh)
    cor = cv2.matchTemplate(
        fm.astype(np.float32), fm_pad.astype(np.float32), cv2.TM_SQDIFF_NORMED
    )
    conts = cv2.findContours(
        (cor < 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    if len(conts) != 1:
        return 0
    cont = conts[0]
    perimeter = cv2.arcLength(cont, True)
    if perimeter <= 0:
        return 0
    area = cv2.contourArea(cont)
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return circularity


def apply_shifts(varr, shifts, fill=np.nan):
    """
    Apply the shifts to the input frames

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        shifts (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the shifts with dimensions: frame, height and width.

    Returns:
        (xarray.DataArray): xarray.DataArray of the shifted input frames (varr)
    """
    sh_dim = shifts.coords["shift_dim"].values.tolist()
    varr_sh = xr.apply_ufunc(
        shift_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        shifts,
        input_core_dims=[sh_dim, ["shift_dim"]],
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


def get_mask(fm, bin_thres, bin_wnd):
    return cv2.adaptiveThreshold(
        fm,
        1,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        bin_wnd,
        -bin_thres,
    ).astype(bool)


def apply_transform(varr, trans, fill=0, mesh_size=None):
    sh_dim = trans.coords["shift_dim"].values.tolist()
    if trans.ndim > 2:
        fm0 = varr.isel(frame=0).values
        if mesh_size is None:
            mesh_size = get_mesh_size(fm0)
        param = get_bspline_param(fm0, mesh_size)
        mdim = ["shift_dim", "grid0", "grid1"]
    else:
        param = None
        mdim = ["shift_dim"]
    varr_sh = xr.apply_ufunc(
        transform_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        trans,
        input_core_dims=[sh_dim, mdim],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask="parallelized",
        kwargs={"fill": fill, "param": param},
        output_dtypes=[varr.dtype],
    )
    return varr_sh


def transform_perframe(fm, tx_coef, fill=0, param=None, mesh_size=None):
    if tx_coef.ndim > 1:
        if param is None:
            if mesh_size is None:
                mesh_size = get_mesh_size(fm)
            param = get_bspline_param(fm, mesh_size)
        tx = sitk.BSplineTransform([sitk.GetImageFromArray(a) for a in tx_coef])
        tx.SetFixedParameters(param)
    else:
        tx = sitk.TranslationTransform(2, -tx_coef[::-1])
    fm = sitk.GetImageFromArray(fm)
    fm = sitk.Resample(fm, fm, tx, sitk.sitkLinear, fill)
    return sitk.GetArrayFromImage(fm)


def get_bspline_param(img, mesh_size):
    return sitk.BSplineTransformInitializer(
        image1=sitk.GetImageFromArray(img), transformDomainMeshSize=mesh_size
    ).GetFixedParameters()


def get_mesh_size(fm):
    return (int(np.around(fm.shape[0] / 100)), int(np.around(fm.shape[1] / 100)))
