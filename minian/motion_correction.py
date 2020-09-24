import cv2
import dask.array as darr
import numpy as np
import xarray as xr
from dask import delayed


def estimate_shifts(varr, max_sh, dim="frame", npart=3, local=False):
    varr = varr.chunk({"height": -1, "width": -1})
    vmax, sh = xr.apply_ufunc(
        est_sh_part,
        varr,
        input_core_dims=[[dim, "height", "width"]],
        output_core_dims=[["height", "width"], [dim, "variable"]],
        vectorize=True,
        dask="allowed",
        kwargs={"max_sh": max_sh, "npart": npart, "local": local},
    )
    return sh.assign_coords(variable=["height", "width"])


def est_sh_part(varr, max_sh, npart, local):
    if varr.shape[0] <= 1:
        return varr.squeeze(), np.array([[0, 0]])
    idx_spt = np.array_split(np.arange(varr.shape[0]), npart)
    fm_ls, sh_ls = [], []
    for idx in idx_spt:
        if len(idx) > 0:
            fm, sh = est_sh_part(varr[idx, :, :], max_sh, npart, local)
            fm_ls.append(fm)
            sh_ls.append(sh)
    mid = int(len(sh_ls) / 2)
    sh_add_ls = [np.array([0, 0])] * len(sh_ls)
    for i, fm in enumerate(fm_ls):
        if i < mid:
            temp = fm_ls[i + 1]
            sh_idx = np.arange(i + 1)
        elif i > mid:
            temp = fm_ls[i - 1]
            sh_idx = np.arange(i, len(sh_ls))
        else:
            continue
        sh_add = darr.from_delayed(
            delayed(match_temp)(fm, temp, max_sh, local), (2,), float
        )
        for j in sh_idx:
            sh_ls[j] = sh_ls[j] + sh_add.reshape((1, -1))
            sh_add_ls[j] = sh_add_ls[j] + sh_add
    for i, (fm, sh) in enumerate(zip(fm_ls, sh_add_ls)):
        fm_ls[i] = darr.nan_to_num(
            darr.from_delayed(delayed(shift_perframe)(fm, sh), fm.shape, fm.dtype)
        )
    sh_ret = darr.concatenate(sh_ls)
    fm_ret = darr.stack(fm_ls)
    return fm_ret.max(axis=0), sh_ret


def match_temp(src, dst, max_sh, local, subpixel=False):
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
