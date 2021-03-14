import functools as fct
import os
import warnings

import cv2
import cvxpy as cvx
import dask as da
import dask.array as darr
import networkx as nx
import numba as nb
import numpy as np
import pandas as pd
import pyfftw.interfaces.numpy_fft as numpy_fft
import pymetis
import scipy.sparse
import sparse
import xarray as xr
import zarr
from distributed import get_client
from scipy.linalg import lstsq, toeplitz
from scipy.ndimage import label
from scipy.signal import butter, lfilter, welch
from scipy.sparse import dia_matrix
from skimage import morphology as moph
from sklearn.linear_model import LassoLars
from statsmodels.tsa.stattools import acovf

from .utilities import (
    custom_arr_optimize,
    custom_delay_optimize,
    open_minian,
    rechunk_like,
    save_minian,
)


def get_noise_fft(varr, noise_range=(0.25, 0.5), noise_method="logmexp"):
    """
    Estimates the power spectral density of the noise within the range specified by the input argument noise_range

    Args:
        varr (type): input data
        noise_range (tuple, optional): noise range. Defaults to (0.25, 0.5).
        noise_method (str, optional): methods for the identification of the principal harmonic in the noise spectrum. Defaults to 'logmexp'.

    Returns:
        type: spectral density of the noise

    """
    try:
        clt = get_client()
        threads = min(clt.nthreads().values())
    except ValueError:
        threads = 1
    sn = xr.apply_ufunc(
        _noise_fft,
        varr,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(
            noise_range=noise_range, noise_method=noise_method, threads=threads
        ),
        output_dtypes=[np.float],
    )
    return sn


def _noise_fft(px, noise_range=(0.25, 0.5), noise_method="logmexp", threads=1):
    """
    Estimates the periodic components in the noise by estimating the power spectral density within the user-defined noise range (noise_range). Noise_range should be set by the user to include the most obvious sources of periodic artifacts (e.g. motion due to heart beat, respiration, etc..). The function is constrained to compute only the non-negative frequency terms of signal (numpy.fft.rfft).

    Args:
        px (type): input data
        noise_range (tuple, optional): Defaults to (0.25, 0.5).
        noise_method (str, optional): Defaults to 'logmexp'.

    Returns:
        type: noise spectrum

    """
    _T = len(px)
    nr = np.around(np.array(noise_range) * _T).astype(int)
    px = 1 / _T * np.abs(numpy_fft.rfft(px, threads=threads)[nr[0] : nr[1]]) ** 2
    if noise_method == "mean":
        return np.sqrt(px.mean())
    elif noise_method == "median":
        return np.sqrt(px.median())
    elif noise_method == "logmexp":
        eps = np.finfo(px.dtype).eps
        return np.sqrt(np.exp(np.log(px + eps).mean()))
    elif noise_method == "sum":
        return np.sqrt(px.sum())


def get_noise_welch(
    varr, noise_range=(0.25, 0.5), noise_method="logmexp", compute=True
):
    print("estimating noise")
    sn = xr.apply_ufunc(
        noise_welch,
        varr.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(noise_range=noise_range, noise_method=noise_method),
        output_dtypes=[varr.dtype],
    )
    if compute:
        sn = sn.compute()
    return sn


def noise_welch(y, noise_range, noise_method):
    """
    Alternative (w.r.t. FFT) formulation to estimate the spectral density of a signal, it assumes the noise in the signal to be a stochastic process (unlike FFT) and attenuates noise by windowing the original signal into segments and averaging over them.

    Args:
        y (type): input
        noise_range (type): noise range
        noise_method (type): method for estimating the principal harmonic in the noise range

    Returns:
        [type]: [noise spectrum]

    """
    ff, Pxx = welch(y)
    mask0, mask1 = ff > noise_range[0], ff < noise_range[1]
    mask = np.logical_and(mask0, mask1)
    Pxx_ind = Pxx[mask]
    sn = {
        "mean": lambda x: np.sqrt(np.mean(x / 2)),
        "median": lambda x: np.sqrt(np.median(x / 2)),
        "logmexp": lambda x: np.sqrt(np.exp(np.mean(np.log(x / 2)))),
    }[noise_method](Pxx_ind)
    return sn


def update_spatial(
    Y,
    A,
    b,
    C,
    f,
    sn,
    dl_wnd=5,
    sparse_penal=0.5,
    update_background=True,
    normalize=True,
    size_thres=(9, None),
    in_memory=False,
):
    """
    This function recursively uses the spatial noise and seed parameters to calculate the signal dynamics across the pixels.

    Args:
        Y (xarray.DataArray): input data
        A (xarray.DataArray): spatial footprint of the initial seeds
        b (xarray.DataArray): spatial footprint of the background
        C (xarray.DataArray): temporal dynamics of cells’ activity
        f (xarray.DataArray): temporal dynamics of the background
        sn (float): spectral density of noise
        gs_sigma (int, optional): Defaults to 6.
        dl_wnd (int, optional): window size that set the maximum dilation of A_init. Defaults to 5.
        sparse_penal (float, optional): sparseness penalty, it provides balance between fidelity to cells’ shape and sparsity of identified cells. Defaults to 0.5.
        update_background (bool, optional): True or False, determines whether the background is updated. Defaults to True.
        post_scal (bool, optional): deprecated.
        normalize (bool, optional): Whether to normalize the resulting spatial footprints so that the sum of coefficients in each footprint is 1. Defaults to True.
        zero_thres (str, optional): The threshold below which the values in the spatial footprints will be set to zero. Defaults to 'eps', which will be interpreted as the machine epsilon of the datatype.
        sched (str, optional): deprecated

    Returns:
        List(xarray.DataArray):

    """
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if in_memory:
        C_store = C.compute().values
    else:
        C_path = os.path.join(intpath, C.name + ".zarr", C.name)
        C_store = zarr.open_array(C_path)
    print("estimating penalty parameter")
    alpha = sparse_penal * sn
    alpha = rechunk_like(alpha.compute(), sn)
    print("computing subsetting matrix")
    selem = moph.disk(dl_wnd)
    sub = xr.apply_ufunc(
        cv2.dilate,
        A,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        kwargs=dict(kernel=selem),
        dask="parallelized",
        output_dtypes=[A.dtype],
    )
    sub = sub > 0
    sub.data = sub.data.map_blocks(sparse.COO)
    if update_background:
        b_in = rechunk_like(b > 0, Y).assign_coords(unit_id=-1).expand_dims("unit_id")
        b_in.data = b_in.data.map_blocks(sparse.COO)
        b_in = b_in.compute()
        sub = xr.concat([sub, b_in], "unit_id")
        f_in = f.compute().data
    else:
        f_in = None
    sub = sub.transpose("height", "width", "unit_id").compute()
    sub.data = darr.array(sub.data).rechunk(
        (Y.data.chunksize[1], Y.data.chunksize[2], -1)
    )
    print("fitting spatial matrix")
    ssub = darr.map_blocks(
        sps_any,
        sub.data,
        drop_axis=2,
        chunks=((1, 1)),
        meta=sparse.ones(1).astype(bool),
    ).compute()
    Y_trans = Y.transpose("height", "width", "frame")
    # take fast route if a lot of chunks are empty
    if ssub.sum() < 500:
        A_new = np.empty(sub.data.numblocks, dtype=object)
        for (hblk, wblk), has_unit in np.ndenumerate(ssub):
            cur_sub = sub.data.blocks[hblk, wblk, :]
            if has_unit:
                cur_blk = update_spatial_block(
                    Y_trans.data.blocks[hblk, wblk, :],
                    alpha.data.blocks[hblk, wblk],
                    cur_sub,
                    C_store=C_store,
                    f=f_in,
                )
            else:
                cur_blk = darr.array(sparse.zeros((cur_sub.shape)))
            A_new[hblk, wblk, 0] = cur_blk
        A_new = darr.block(A_new.tolist())
    else:
        A_new = update_spatial_block(
            Y_trans.data, alpha.data, sub.data, C_store=C_store, f=f_in,
        )
    with da.config.set(**{"optimization.fuse.ave-width": 6}):
        A_new = da.optimize(A_new)[0]
    if normalize:
        A_new = A_new / A_new.sum(axis=(0, 1))
    A_new = xr.DataArray(
        darr.moveaxis(A_new, -1, 0).map_blocks(lambda a: a.todense(), dtype=A.dtype),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": sub.coords["unit_id"],
            "height": A.coords["height"],
            "width": A.coords["width"],
        },
    )
    A_new = save_minian(
        A_new.rename("A_new"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    if update_background:
        print("updating background")
        b_new = A_new.sel(unit_id=-1)
        f_new = xr.apply_ufunc(
            darr.tensordot,
            Y,
            b_new,
            input_core_dims=[["frame", "height", "width"], ["height", "width"]],
            output_core_dims=[["frame"]],
            kwargs={"axes": [(1, 2), (0, 1)]},
            dask="allowed",
        )
        b_new = b_new.compute()
        f_new = f_new.compute()
        A_new = A_new[:-1, :, :]
    else:
        b_new = b.compute()
        f_new = f.compute()
    if size_thres:
        low, high = size_thres
        A_bin = A_new > 0
        mask = np.ones(A_new.sizes["unit_id"], dtype=bool)
        if low:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) > low).compute(), mask
            )
        if high:
            mask = np.logical_and(
                (A_bin.sum(["height", "width"]) < high).compute(), mask
            )
        mask = xr.DataArray(
            mask, dims=["unit_id"], coords={"unit_id": A_new.coords["unit_id"].values}
        )
    else:
        mask = (A_new.sum(["height", "width"]) > 0).compute()
    print("{} out of {} units dropped".format(len(mask) - mask.sum().values, len(mask)))
    A_new = A_new.sel(unit_id=mask)
    return A_new, b_new, f_new, mask


def sps_any(x):
    return np.atleast_2d(x.nnz > 0)


def update_spatial_perpx(y, alpha, sub, C_store, f):
    if f is not None:
        idx = sub[:-1].nonzero()[0]
    else:
        idx = sub.nonzero()[0]
    try:
        C = C_store.get_orthogonal_selection((idx, slice(None))).T
    except AttributeError:
        C = C_store[idx, :].T
    if (f is not None) and sub[-1]:
        C = np.concatenate([C, f.reshape((-1, 1))], axis=1)
        idx = np.concatenate([idx, np.array(len(sub) - 1).reshape(-1)])
    clf = LassoLars(alpha=alpha, positive=True)
    coef = clf.fit(C, y).coef_
    mask = coef > 0
    coef = coef[mask]
    idx = idx[mask]
    return sparse.COO(coords=idx, data=coef, shape=sub.shape)


@darr.as_gufunc(signature="(f),(),(u)->(u)", output_dtypes=float)
def update_spatial_block(y, alpha, sub, **kwargs):
    C_store = kwargs.get("C_store")
    f = kwargs.get("f")
    crd_ls = []
    data_ls = []
    for h, w in zip(*sub.any(axis=-1).nonzero()):
        res = update_spatial_perpx(y[h, w, :], alpha[h, w], sub[h, w, :], C_store, f)
        crd = res.coords
        crd = np.concatenate([np.full_like(crd, h), np.full_like(crd, w), crd], axis=0)
        crd_ls.append(crd)
        data_ls.append(res.data)
    if data_ls:
        return sparse.COO(
            coords=np.concatenate(crd_ls, axis=1),
            data=np.concatenate(data_ls),
            shape=sub.shape,
        )
    else:
        return sparse.zeros(sub.shape)


def compute_trace(Y, A, b, C, f):
    fms = Y.coords["frame"]
    uid = A.coords["unit_id"]
    Y = Y.data
    A = darr.from_array(A.data.map_blocks(sparse.COO).compute(), chunks=-1)
    C = C.data.map_blocks(sparse.COO).T
    b = (
        b.fillna(0)
        .data.map_blocks(sparse.COO)
        .reshape((1, Y.shape[1], Y.shape[2]))
        .compute()
    )
    f = f.fillna(0).data.reshape((-1, 1))
    AtA = darr.tensordot(A, A, axes=[(1, 2), (1, 2)]).compute()
    A_norm = (
        (1 / (A ** 2).sum(axis=(1, 2)))
        .map_blocks(
            lambda a: sparse.diagonalize(sparse.COO(a)), chunks=(A.shape[0], A.shape[0])
        )
        .compute()
    )
    B = darr.tensordot(f, b, axes=[(1), (0)])
    Y = Y - B
    YtA = darr.tensordot(Y, A, axes=[(1, 2), (1, 2)])
    YtA = darr.dot(YtA, A_norm)
    CtA = darr.dot(C, AtA)
    CtA = darr.dot(CtA, A_norm)
    YrA = (YtA - CtA + C).clip(0)
    arr_opt = fct.partial(
        custom_arr_optimize,
        inline_patterns=["from-getitem-transpose"],
        rename_dict={"tensordot": "tensordot_restricted"},
    )
    with da.config.set(array_optimize=arr_opt):
        YrA = da.optimize(YrA)[0]
    YrA = xr.DataArray(
        YrA, dims=["frame", "unit_id"], coords={"frame": fms, "unit_id": uid},
    )
    return YrA.transpose("unit_id", "frame")


def update_temporal(
    A,
    C,
    b=None,
    f=None,
    Y=None,
    YrA=None,
    noise_freq=0.25,
    p=2,
    add_lag="p",
    jac_thres=0.1,
    sparse_penal=1,
    bseg=None,
    zero_thres=1e-8,
    max_iters=200,
    use_smooth=True,
    normalize=True,
    warm_start=False,
    post_scal=True,
    scs_fallback=False,
    concurrent_update=False,
):
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if YrA is None:
        YrA = compute_trace(Y, A, b, C, f).persist()
    Ymask = (YrA > 0).any("frame").compute()
    A, C, YrA = A.sel(unit_id=Ymask), C.sel(unit_id=Ymask), YrA.sel(unit_id=Ymask)
    print("grouping overlaping units")
    A_sps = (A.data.map_blocks(sparse.COO) > 0).compute().astype(np.float32)
    A_inter = sparse.tensordot(A_sps, A_sps, axes=[(1, 2), (1, 2)])
    A_usum = np.tile(A_sps.sum(axis=(1, 2)).todense(), (A_sps.shape[0], 1))
    A_usum = A_usum + A_usum.T
    jac = scipy.sparse.csc_matrix(A_inter / (A_usum - A_inter) > jac_thres)
    unit_labels = label_connected(jac)
    YrA = YrA.assign_coords(unit_labels=("unit_id", unit_labels))
    print("updating temporal components")
    c_ls = []
    s_ls = []
    b_ls = []
    c0_ls = []
    g_ls = []
    uid_ls = []
    grp_dim = "unit_labels"
    C = C.assign_coords(unit_labels=("unit_id", unit_labels))
    if warm_start:
        C.data = C.data.map_blocks(scipy.sparse.csr_matrix)
    inline_opt = fct.partial(
        custom_delay_optimize, inline_patterns=["getitem", "rechunk-merge"],
    )
    for cur_YrA, cur_C in zip(YrA.groupby(grp_dim), C.groupby(grp_dim)):
        uid_ls.append(cur_YrA[1].coords["unit_id"].values.reshape(-1))
        cur_YrA, cur_C = cur_YrA[1].data.rechunk(-1), cur_C[1].data.rechunk(-1)
        # memory demand for cvxpy is roughly 500 times input
        mem_demand = cur_YrA.nbytes / 1e6 * 500
        # issue a warning if expected memory demand is larger than 1G
        if concurrent_update and (mem_demand > 1e3):
            warnings.warn(
                "{} MB of memory is expected for concurrent update, "
                "which might be too demanding. "
                "Consider merging the units "
                "or changing jaccard threshold".format(cur_YrA.shape[0])
            )
        if not warm_start:
            cur_C = None
        if cur_YrA.shape[0] > 1:
            dl_opt = inline_opt
        else:
            dl_opt = custom_delay_optimize
        # explicitly using delay (rather than gufunc) seem to promote the
        # depth-first behavior of dask
        with da.config.set(delayed_optimize=dl_opt):
            res = da.optimize(
                da.delayed(update_temporal_block)(
                    cur_YrA,
                    noise_freq=noise_freq,
                    p=p,
                    add_lag=add_lag,
                    normalize=normalize,
                    concurrent=concurrent_update,
                    use_smooth=use_smooth,
                    c_last=cur_C,
                    bseg=bseg,
                    sparse_penal=sparse_penal,
                    max_iters=max_iters,
                    scs_fallback=scs_fallback,
                    zero_thres=zero_thres,
                )
            )[0]
        c_ls.append(darr.from_delayed(res[0], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        s_ls.append(darr.from_delayed(res[1], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        b_ls.append(darr.from_delayed(res[2], shape=cur_YrA.shape, dtype=cur_YrA.dtype))
        c0_ls.append(
            darr.from_delayed(res[3], shape=cur_YrA.shape, dtype=cur_YrA.dtype)
        )
        g_ls.append(
            darr.from_delayed(res[4], shape=(cur_YrA.shape[0], p), dtype=cur_YrA.dtype)
        )
    uids_new = np.concatenate(uid_ls)
    C_new = xr.DataArray(
        darr.concatenate(c_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={"unit_id": uids_new, "frame": YrA.coords["frame"],},
        name="C_new",
    )
    S_new = xr.DataArray(
        darr.concatenate(s_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={"unit_id": uids_new, "frame": YrA.coords["frame"].values,},
        name="S_new",
    )
    b0_new = xr.DataArray(
        darr.concatenate(b_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={"unit_id": uids_new, "frame": YrA.coords["frame"].values,},
        name="b0_new",
    )
    c0_new = xr.DataArray(
        darr.concatenate(c0_ls, axis=0),
        dims=["unit_id", "frame"],
        coords={"unit_id": uids_new, "frame": YrA.coords["frame"].values,},
        name="c0_new",
    )
    g = xr.DataArray(
        darr.concatenate(g_ls, axis=0),
        dims=["unit_id", "lag"],
        coords={"unit_id": uids_new, "lag": np.arange(p)},
        name="g",
    )
    arr_opt = fct.partial(custom_arr_optimize, keep_patterns=["^update_temporal_block"])
    with da.config.set(array_optimize=arr_opt):
        da.compute(
            [
                save_minian(
                    var.chunk({"unit_id": 1}), intpath, compute=False, overwrite=True
                )
                for var in [C_new, S_new, b0_new, c0_new, g]
            ]
        )
    int_ds = open_minian(intpath, return_dict=True)
    C_new, S_new, b0_new, c0_new, g = (
        int_ds["C_new"],
        int_ds["S_new"],
        int_ds["b0_new"],
        int_ds["c0_new"],
        int_ds["g"],
    )
    mask = (S_new.sum("frame") > 0).compute()
    print("{} out of {} units dropped".format((~mask).sum().values, len(Ymask)))
    C_new, S_new, b0_new, c0_new, g = (
        C_new[mask],
        S_new[mask],
        b0_new[mask],
        c0_new[mask],
        g[mask],
    )
    sig_new = C_new + b0_new + c0_new
    YrA_new = YrA.sel(unit_id=mask)
    if post_scal and len(sig_new) > 0:
        print("post-hoc scaling")
        scal = (
            lstsq_vec(sig_new.data.rechunk((1, -1)), YrA_new.data)
            .compute()
            .reshape((-1, 1))
        )
        C_new, S_new, b0_new, c0_new = (
            C_new * scal,
            S_new * scal,
            b0_new * scal,
            c0_new * scal,
        )
    return C_new, S_new, b0_new, c0_new, g, mask


@darr.as_gufunc(signature="(f),(f)->()", output_dtypes=float)
def lstsq_vec(a, b):
    a = a.reshape((-1, 1))
    return np.linalg.lstsq(a, b.squeeze(), rcond=-1)[0]


def get_ar_coef(y, sn, p, add_lag, pad=None):
    if add_lag == "p":
        max_lag = p * 2
    else:
        max_lag = p + add_lag
    cov = acovf(y, fft=True)
    C_mat = toeplitz(cov[:max_lag], cov[:p]) - sn ** 2 * np.eye(max_lag, p)
    g = lstsq(C_mat, cov[1 : max_lag + 1])[0]
    if pad:
        res = np.zeros(pad)
        res[: len(g)] = g
        return res
    else:
        return g


def get_p(y):
    dif = np.append(np.diff(y), 0)
    rising = dif > 0
    prd_ris, num_ris = label(rising)
    ext_prd = np.zeros(num_ris)
    for id_prd in range(num_ris):
        prd = y[prd_ris == id_prd + 1]
        ext_prd[id_prd] = prd[-1] - prd[0]
    id_max_prd = np.argmax(ext_prd)
    return np.sum(rising[prd_ris == id_max_prd + 1])


def update_temporal_block(
    YrA,
    noise_freq,
    p=None,
    add_lag="p",
    normalize=True,
    use_smooth=True,
    concurrent=False,
    **kwargs
):
    vec_get_noise = np.vectorize(
        _noise_fft,
        otypes=[float],
        excluded=["noise_range", "noise_method"],
        signature="(f)->()",
    )
    vec_get_p = np.vectorize(get_p, otypes=[int], signature="(f)->()")
    vec_get_ar_coef = np.vectorize(
        get_ar_coef,
        otypes=[float],
        excluded=["pad", "add_lag"],
        signature="(f),(),()->(l)",
    )
    if normalize:
        amean = YrA.sum(axis=1).mean()
        YrA /= amean
        YrA *= YrA.shape[1]
    tn = vec_get_noise(YrA, noise_range=(noise_freq, 1))
    if use_smooth:
        YrA_ar = filt_fft_vec(YrA, noise_freq, "low")
        tn_ar = vec_get_noise(YrA_ar, noise_range=(noise_freq, 1))
    else:
        YrA_ar, tn_ar = YrA, tn
    # auto estimation of p is disabled since it's never used and makes it
    # impossible to pre-determine the shape of output
    # if p is None:
    #     p = np.clip(vec_get_p(YrA_ar), 1, None)
    pmax = np.max(p)
    g = vec_get_ar_coef(YrA_ar, tn_ar, p, pad=pmax, add_lag=add_lag)
    del YrA_ar, tn_ar
    if concurrent:
        c, s, b, c0 = update_temporal_cvxpy(YrA, g, tn, **kwargs)
    else:
        res_ls = []
        for cur_yra, cur_g, cur_tn in zip(YrA, g, tn):
            res = update_temporal_cvxpy(cur_yra, cur_g, cur_tn, **kwargs)
            res_ls.append(res)
        c = np.concatenate([r[0] for r in res_ls], axis=0)
        s = np.concatenate([r[1] for r in res_ls], axis=0)
        b = np.concatenate([r[2] for r in res_ls], axis=0)
        c0 = np.concatenate([r[3] for r in res_ls], axis=0)
    return c, s, b, c0, g


def update_temporal_cvxpy(y, g, sn, A=None, bseg=None, **kwargs):
    # spatial:
    # (d, f), (u, p), (d), (d, u)
    # (d, f), (p), (d), (d)
    # trace:
    # (u, f), (u, p), (u)
    # (f), (p), ()

    # get_parameters
    sparse_penal = kwargs.get("sparse_penal")
    max_iters = kwargs.get("max_iters")
    use_cons = kwargs.get("use_cons", False)
    scs = kwargs.get("scs_fallback")
    c_last = kwargs.get("c_last")
    zero_thres = kwargs.get("zero_thres")
    # conform variables to generalize multiple unit case
    if y.ndim < 2:
        y = y.reshape((1, -1))
    if g.ndim < 2:
        g = g.reshape((1, -1))
    sn = np.atleast_1d(sn)
    if A is not None:
        if A.ndim < 2:
            A = A.reshape((-1, 1))
    # get count of frames and units
    _T = y.shape[-1]
    _u = g.shape[0]
    if A is not None:
        _d = A.shape[0]
    # construct G matrix and decay vector per unit
    dc_vec = np.zeros((_u, _T))
    G_ls = []
    for cur_u in range(_u):
        cur_g = g[cur_u, :]
        # construct first column and row
        cur_c = np.zeros(_T)
        cur_c[0] = 1
        cur_c[1 : len(cur_g) + 1] = -cur_g
        # update G with toeplitz matrix
        G_ls.append(
            cvx.Constant(
                dia_matrix(
                    (
                        np.tile(np.concatenate(([1], -cur_g)), (_T, 1)).T,
                        -np.arange(len(cur_g) + 1),
                    ),
                    shape=(_T, _T),
                ).tocsc()
            )
        )
        # update dc_vec
        cur_gr = np.roots(cur_c)
        dc_vec[cur_u, :] = np.max(cur_gr.real) ** np.arange(_T)
    # get noise threshold
    thres_sn = sn * np.sqrt(_T)
    # construct variables
    if bseg is not None:
        nseg = int(np.max(bseg) + 1)
        b_temp = np.zeros((nseg, _T))
        for iseg in range(nseg):
            b_temp[iseg, bseg == iseg] = 1
        b_cmp = cvx.Variable((_u, nseg))
    else:
        b_temp = np.ones((1, _T))
        b_cmp = cvx.Variable((_u, 1))
    b = b_cmp @ b_temp  # baseline fluorescence per unit
    c0 = cvx.Variable(_u)  # initial fluorescence per unit
    c = cvx.Variable((_u, _T))  # calcium trace per unit
    if c_last is not None:
        c.value = c_last
        warm_start = True
    else:
        warm_start = False
    s = cvx.vstack([G_ls[u] @ c[u, :] for u in range(_u)])  # spike train per unit
    # residual noise per unit
    if A is not None:
        sig = cvx.vstack(
            [
                (A * c)[px, :] + (A * b)[px, :] + (A * cvx.diag(c0) * dc_vec)[px, :]
                for px in range(_d)
            ]
        )
        noise = y - sig
    else:
        sig = cvx.vstack([c[u, :] + b[u, :] + c0[u] * dc_vec[u, :] for u in range(_u)])
        noise = y - sig
    noise = cvx.vstack([cvx.norm(noise[i, :], 2) for i in range(noise.shape[0])])
    # construct constraints
    cons = []
    cons.append(
        b >= np.broadcast_to(np.min(y, axis=-1).reshape((-1, 1)), y.shape)
    )  # baseline larger than minimum
    cons.append(c0 >= 0)  # initial fluorescence larger than 0
    cons.append(s >= 0)  # spike train non-negativity
    # noise constraints
    cons_noise = [noise[i] <= thres_sn[i] for i in range(thres_sn.shape[0])]
    try:
        obj = cvx.Minimize(cvx.sum(cvx.norm(s, 1, axis=1)))
        prob = cvx.Problem(obj, cons + cons_noise)
        if use_cons:
            _ = prob.solve(solver="ECOS")
        if not (prob.status == "optimal" or prob.status == "optimal_inaccurate"):
            if use_cons:
                warnings.warn("constrained version of problem infeasible")
            raise ValueError
    except (ValueError, cvx.SolverError):
        lam = sn * sparse_penal
        obj = cvx.Minimize(
            cvx.sum(cvx.sum(noise, axis=1) + cvx.multiply(lam, cvx.norm(s, 1, axis=1)))
        )
        prob = cvx.Problem(obj, cons)
        try:
            _ = prob.solve(solver="ECOS", warm_start=warm_start, max_iters=max_iters)
            if prob.status in ["infeasible", "unbounded", None]:
                raise ValueError
        except (cvx.SolverError, ValueError):
            try:
                if scs:
                    _ = prob.solve(solver="SCS", max_iters=200)
                if prob.status in ["infeasible", "unbounded", None]:
                    raise ValueError
            except (cvx.SolverError, ValueError):
                warnings.warn(
                    "problem status is {}, returning zero".format(prob.status),
                    RuntimeWarning,
                )
                return [np.zeros(c.shape, dtype=float)] * 4
    if not (prob.status == "optimal"):
        warnings.warn("problem solved sub-optimally", RuntimeWarning)
    c = np.where(c.value > zero_thres, c.value, 0)
    s = np.where(s.value > zero_thres, s.value, 0)
    b = np.where(b.value > zero_thres, b.value, 0)
    c0 = c0.value.reshape((-1, 1)) * dc_vec
    c0 = np.where(c0 > zero_thres, c0, 0)
    return c, s, b, c0


def unit_merge(A, C, add_list=None, thres_corr=0.9, noise_freq=None):
    print("computing spatial overlap")
    with da.config.set(
        array_optimize=darr.optimization.optimize,
        **{"optimization.fuse.subgraphs": False}
    ):
        A_sps = (A.data.map_blocks(sparse.COO) > 0).rechunk(-1).persist()
        A_inter = sparse.tril(
            darr.tensordot(
                A_sps.astype(np.float32),
                A_sps.astype(np.float32),
                axes=[(1, 2), (1, 2)],
            ).compute(),
            k=-1,
        )
    print("computing temporal correlation")
    nod_df = pd.DataFrame({"unit_id": A.coords["unit_id"].values})
    adj = adj_corr(C, A_inter, nod_df, freq=noise_freq)
    print("labeling units to be merged")
    adj = adj > thres_corr
    adj = adj + adj.T
    unit_labels = xr.apply_ufunc(
        label_connected,
        adj,
        input_core_dims=[["unit_id", "unit_id_cp"]],
        output_core_dims=[["unit_id"]],
    )
    print("merging units")
    A_merge = (
        A.assign_coords(unit_labels=("unit_id", unit_labels))
        .groupby("unit_labels")
        .mean("unit_id")
        .rename(unit_labels="unit_id")
    )
    C_merge = (
        C.assign_coords(unit_labels=("unit_id", unit_labels))
        .groupby("unit_labels")
        .mean("unit_id")
        .rename(unit_labels="unit_id")
    )
    if add_list:
        for ivar, var in enumerate(add_list):
            var_mrg = (
                var.assign_coords(unit_labels=("unit_id", unit_labels))
                .groupby("unit_labels")
                .mean("unit_id")
                .rename(unit_labels="unit_id")
            )
            add_list[ivar] = var_mrg
        return A_merge, C_merge, add_list
    else:
        return A_merge, C_merge


def label_connected(adj, only_connected=False):
    try:
        np.fill_diagonal(adj, 0)
        adj = np.triu(adj)
        g = nx.convert_matrix.from_numpy_matrix(adj)
    except:
        g = nx.convert_matrix.from_scipy_sparse_matrix(adj)
    labels = np.zeros(adj.shape[0], dtype=np.int)
    for icomp, comp in enumerate(nx.connected_components(g)):
        comp = list(comp)
        if only_connected and len(comp) == 1:
            labels[comp] = -1
        else:
            labels[comp] = icomp
    return labels


def smooth_sig(sig, freq, method="fft", btype="low"):
    try:
        filt_func = {"fft": filt_fft, "butter": filt_butter}[method]
    except KeyError:
        raise NotImplementedError(method)
    sig_smth = xr.apply_ufunc(
        filt_func,
        sig,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        kwargs={"btype": btype, "freq": freq},
        dask="parallelized",
        output_dtypes=[sig.dtype],
    )
    return sig_smth


def filt_fft(x, freq, btype):
    _T = len(x)
    if btype == "low":
        zero_range = slice(int(freq * _T), None)
    elif btype == "high":
        zero_range = slice(None, int(freq * _T))
    xfft = numpy_fft.rfft(x)
    xfft[zero_range] = 0
    return numpy_fft.irfft(xfft, len(x))


def filt_butter(x, freq, btype):
    but_b, but_a = butter(2, freq * 2, btype=btype, analog=False)
    return lfilter(but_b, but_a, x)


def filt_fft_vec(x, freq, btype):
    for ix, xx in enumerate(x):
        x[ix, :] = filt_fft(xx, freq, btype)
    return x


def compute_AtC(A, C):
    fm, h, w = (
        C.coords["frame"].values,
        A.coords["height"].values,
        A.coords["width"].values,
    )
    A = darr.from_array(
        A.data.map_blocks(sparse.COO, dtype=A.dtype).compute(), chunks=-1
    )
    C = C.transpose("frame", "unit_id").data.map_blocks(sparse.COO, dtype=C.dtype)
    AtC = darr.tensordot(C, A, axes=(1, 0)).map_blocks(
        lambda a: a.todense(), dtype=A.dtype
    )
    return xr.DataArray(
        AtC,
        dims=["frame", "height", "width"],
        coords={"frame": fm, "height": h, "width": w},
    )


def graph_optimize_corr(
    varr, G, freq, idx_dims=["height", "width"], chunk=600, step_size=50
):
    # a heuristic to make number of partitions scale with nodes
    n_cuts, membership = pymetis.part_graph(
        int(G.number_of_nodes() / chunk), adjacency=adj_list(G)
    )
    nx.set_node_attributes(
        G, {k: {"part": v} for k, v in zip(sorted(G.nodes), membership)}
    )
    eg_df = nx.to_pandas_edgelist(G)
    part_map = nx.get_node_attributes(G, "part")
    eg_df["part_src"] = eg_df["source"].map(part_map)
    eg_df["part_tgt"] = eg_df["target"].map(part_map)
    eg_df["part_diff"] = (eg_df["part_src"] - eg_df["part_tgt"]).astype(bool)
    corr_ls = []
    idx_ls = []
    npxs = []
    egd_same, egd_diff = eg_df[~eg_df["part_diff"]], eg_df[eg_df["part_diff"]]
    idx_dict = {d: nx.get_node_attributes(G, d) for d in idx_dims}

    def construct_comput(edf, pxs):
        px_map = {k: v for v, k in enumerate(pxs)}
        ridx = edf["source"].map(px_map).values
        cidx = edf["target"].map(px_map).values
        idx_arr = {
            d: xr.DataArray([dd[p] for p in pxs], dims="pixels")
            for d, dd in idx_dict.items()
        }
        vsub = varr.sel(**idx_arr).data
        if len(idx_arr) > 1:  # vectorized indexing
            vsub = vsub.T
        else:
            vsub = vsub.rechunk(-1)
        with da.config.set(**{"optimization.fuse.ave-width": vsub.shape[0]}):
            return da.optimize(smooth_corr(vsub, ridx, cidx, freq=freq))[0]

    for _, eg_sub in egd_same.groupby("part_src"):
        pixels = list(set(eg_sub["source"]) | set(eg_sub["target"]))
        corr_ls.append(construct_comput(eg_sub, pixels))
        idx_ls.append(eg_sub.index)
        npxs.append(len(pixels))
    pixels = set()
    eg_ls = []
    for _, eg_sub in egd_diff.sort_values("source").groupby(
        np.arange(len(egd_diff)) // step_size
    ):
        pixels = pixels | set(eg_sub["source"]) | set(eg_sub["target"])
        eg_ls.append(eg_sub)
        if len(pixels) > chunk - step_size / 2:
            pixels = list(pixels)
            edf = pd.concat(eg_ls)
            corr_ls.append(construct_comput(edf, pixels))
            idx_ls.append(edf.index)
            npxs.append(len(pixels))
            pixels = set()
            eg_ls = []
    print("pixel recompute ratio: {}".format(sum(npxs) / G.number_of_nodes()))
    print("computing correlations")
    corr_ls = da.compute(corr_ls)[0]
    corr = pd.Series(np.concatenate(corr_ls), index=np.concatenate(idx_ls), name="corr")
    eg_df["corr"] = corr
    return eg_df


def adj_corr(varr, adj, nod_df, freq):
    G = nx.Graph()
    G.add_nodes_from([(i, d) for i, d in enumerate(nod_df.to_dict("records"))])
    G.add_edges_from([(s, t) for s, t in zip(*adj.nonzero())])
    corr_df = graph_optimize_corr(varr, G, freq, idx_dims=nod_df.columns)
    adj = scipy.sparse.csr_matrix(
        (corr_df["corr"], (corr_df["source"], corr_df["target"])), shape=adj.shape
    )
    return adj


def adj_list(G):
    gdict = nx.to_dict_of_dicts(G)
    return [np.array(list(gdict[k].keys())) for k in sorted(gdict.keys())]


@darr.as_gufunc(signature="(p,f),(i),(i)->(i)", output_dtypes=[float])
def smooth_corr(X, ridx, cidx, freq):
    if freq:
        X = filt_fft_vec(X, freq, "low")
    return idx_corr(X, ridx, cidx)


@nb.jit(nopython=True, nogil=True, cache=True)
def idx_corr(X, ridx, cidx):
    res = np.zeros(ridx.shape[0])
    std = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        X[i, :] -= X[i, :].mean()
        std[i] = np.sqrt((X[i, :] ** 2).sum())
    for i, (r, c) in enumerate(zip(ridx, cidx)):
        res[i] = (X[r, :] * X[c, :]).sum() / (std[r] * std[c])
    return res
