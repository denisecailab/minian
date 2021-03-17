import functools as fct
import itertools as itt

import cv2
import dask as da
import dask.array as darr
import numpy as np
import SimpleITK as sitk
import xarray as xr

from .utilities import custom_arr_optimize, xrconcat_recursive


def estimate_motion(varr, mtype, dim="frame", npart=None, temp_nfm=30, **kwargs):
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
            vmax, sh = est_motion_part(va.data, mtype, npart, temp_nfm, **kwargs)
            if mtype == "bspline":
                sh = xr.DataArray(
                    sh,
                    dims=[dim, "shift_dim", "grid0", "grid1"],
                    coords={
                        dim: va.coords[dim].values,
                        "shift_dim": ["width", "height"],
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
        vmax, sh = est_motion_part(varr.data, mtype, npart, temp_nfm, **kwargs)
        if mtype == "bspline":
            sh = xr.DataArray(
                sh,
                dims=[dim, "shift_dim", "grid0", "grid1"],
                coords={
                    dim: varr.coords[dim].values,
                    "shift_dim": ["width", "height"],
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


def est_motion_part(varr, mtype, npart, temp_nfm, **kwargs):
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
    arr_opt = fct.partial(
        custom_arr_optimize, keep_patterns=["^est_sh_chunk", "^est_trans_chunk"]
    )
    est_func = {
        "translation": da.delayed(est_sh_chunk),
        "bspline": da.delayed(est_trans_chunk),
    }[mtype]
    if mtype == "bspline":
        mesh_size = kwargs["mesh_size"]
        if mesh_size is None:
            mesh_size = get_mesh_size(varr[0])
            kwargs["mesh_size"] = mesh_size
        param = get_bspline_param(varr[0].compute(), kwargs["mesh_size"])
    tmp_ls = []
    sh_ls = []
    for blk in varr.blocks:
        res = est_func(blk, None, **kwargs)
        tmp = darr.from_delayed(
            res[0], shape=(blk.shape[1], blk.shape[2]), dtype=blk.dtype
        )
        if mtype == "bspline":
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
            kwargs["bin_thres"] = None
            res = est_func(tmps, sh_org_ls, **kwargs)
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


def est_trans_chunk(varr, trans_org, mesh_size, niter, bin_thres=None, bin_wnd=None):
    if bin_thres:
        masks = []
        for fm in varr:
            masks.append(
                cv2.adaptiveThreshold(
                    fm,
                    1,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    bin_wnd,
                    -bin_thres,
                ).astype(bool)
            )
    else:
        masks = [None] * varr.shape[0]
    mid = int(varr.shape[0] / 2)
    param = get_bspline_param(varr[0], mesh_size)
    # the dimension of transform coef array is: frame, 2, grid_size1, grid_size0
    transform = np.zeros((varr.shape[0], 2, int(param[1]), int(param[0])))
    for i, fm in enumerate(varr):
        if i < mid:
            temp = varr[i + 1]
            tmp_ma = masks[i + 1]
            slc = slice(0, i + 1)
        elif i > mid:
            temp = varr[i - 1]
            tmp_ma = masks[i - 1]
            slc = slice(i, None)
        else:
            continue
        tx = ffd_transform(
            fm, temp, mesh_size, src_ma=masks[i], dst_ma=tmp_ma, niter=niter
        )
        coef = np.stack(
            [sitk.GetArrayFromImage(im) for im in tx.Downcast().GetCoefficientImages()]
        )
        transform[slc] = transform[slc] + coef
    for i, tx_coef in enumerate(transform):
        varr[i] = transform_perframe(varr[i], tx_coef, fill=0, param=param)
    if trans_org is not None:
        transform = np.concatenate(
            [transform[i] + torg for i, torg in enumerate(trans_org)], axis=0
        )
    return varr.max(axis=0), transform


def ffd_transform(src, dst, mesh_size, src_ma=None, dst_ma=None, niter=10):
    src = sitk.GetImageFromArray(src.astype(np.float32))
    dst = sitk.GetImageFromArray(dst.astype(np.float32))
    reg = sitk.ImageRegistrationMethod()
    trans_init = sitk.BSplineTransformInitializer(
        image1=dst, transformDomainMeshSize=mesh_size
    )
    if src_ma is not None:
        reg.SetMetricMovingMask(sitk.GetImageFromArray(src_ma.astype(np.uint8)))
    if dst_ma is not None:
        reg.SetMetricFixedMask(sitk.GetImageFromArray(dst_ma.astype(np.uint8)))
    reg.SetInitialTransform(trans_init)
    reg.SetMetricAsMeanSquares()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, convergenceMinimumValue=1e-5, numberOfIterations=niter
    )
    tx = reg.Execute(dst, src)
    return tx


def apply_transform(varr, trans, fill=0, mesh_size=None):
    sh_dim = trans.coords["shift_dim"].values.tolist()
    fm0 = varr.isel(frame=0).values
    if mesh_size is None:
        mesh_size = get_mesh_size(fm0)
    param = get_bspline_param(fm0, mesh_size)
    varr_sh = xr.apply_ufunc(
        transform_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        trans,
        input_core_dims=[sh_dim, ["shift_dim", "grid0", "grid1"]],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask="parallelized",
        kwargs={"fill": fill, "param": param},
        output_dtypes=[varr.dtype],
    )
    return varr_sh


def transform_perframe(fm, tx_coef, fill=0, param=None, mesh_size=None):
    if param is None:
        if mesh_size is None:
            mesh_size = get_mesh_size(fm)
        param = get_bspline_param(fm, mesh_size)
    fm = sitk.GetImageFromArray(fm)
    tx = sitk.BSplineTransform([sitk.GetImageFromArray(a) for a in tx_coef])
    tx.SetFixedParameters(param)
    fm = sitk.Resample(fm, fm, tx, sitk.sitkLinear, fill)
    return sitk.GetArrayFromImage(fm)


def optimize_bspline_composite(comp_tx):
    coef_ls = []
    for itx in range(comp_tx.GetNumberOfTransforms()):
        tx = comp_tx.GetNthTransform(itx).Downcast()
        try:
            coef_im = tx.GetCoefficientImages()
        except AttributeError:
            return comp_tx
        coef = np.stack([sitk.GetArrayFromImage(ci) for ci in coef_im], axis=0)
        coef_ls.append(coef)
    coef = np.stack(coef_ls, axis=0).sum(axis=0)
    tx_new = sitk.BSplineTransform([sitk.GetImageFromArray(a) for a in coef])
    tx_new.SetFixedParameters(tx.GetFixedParameters())
    return tx_new


def get_bspline_param(img, mesh_size):
    return sitk.BSplineTransformInitializer(
        image1=sitk.GetImageFromArray(img), transformDomainMeshSize=mesh_size
    ).GetFixedParameters()


def get_mesh_size(fm):
    return (int(np.around(fm.shape[0] / 100)), int(np.around(fm.shape[1] / 100)))
