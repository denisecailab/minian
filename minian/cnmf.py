import functools as fct
import warnings

import cv2
import cvxpy as cvx
import dask as da
import dask.array as darr
import networkx as nx
import numpy as np
import xarray as xr
from dask import delayed
from scipy.linalg import lstsq, toeplitz
from scipy.ndimage import label
from scipy.signal import butter, lfilter, welch
from scipy.sparse import dia_matrix, diags
from skimage import morphology as moph
from sklearn.linear_model import LassoLars
from sklearn.utils import parallel_backend
from statsmodels.tsa.stattools import acovf

from .utilities import rechunk_like


def get_noise_fft(varr, noise_range=(0.25, 0.5), noise_method="logmexp"):
    sn = xr.apply_ufunc(
        _noise_fft,
        varr.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(noise_range=noise_range, noise_method=noise_method),
        output_dtypes=[np.float],
    )
    return sn


def _noise_fft(px, noise_range=(0.25, 0.5), noise_method="logmexp"):
    _T = len(px)
    nr = np.around(np.array(noise_range) * 2 * _T).astype(int)
    px_fft = np.fft.rfft(px)
    px_psd = 1 / _T * np.abs(px_fft) ** 2
    px_band = px_psd[nr[0] : nr[1]]
    if noise_method == "mean":
        return np.sqrt(px_band.mean())
    elif noise_method == "median":
        return np.sqrt(px_band.median())
    elif noise_method == "logmexp":
        eps = np.finfo(px_band.dtype).eps
        return np.sqrt(np.exp(np.log(px_band + eps).mean()))
    elif noise_method == "sum":
        return np.sqrt(px_band.sum())


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
    gs_sigma=6,
    dl_wnd=5,
    sparse_penal=0.5,
    update_background=True,
    post_scal=False,
    normalize=True,
    zero_thres="eps",
    sched="single-threaded",
):
    _T = len(Y.coords["frame"])
    print("estimating penalty parameter")
    cct = C.dot(C, "frame")
    alpha = sparse_penal * sn * np.sqrt(np.max(np.diag(cct))) / _T
    alpha = alpha.persist()
    print("computing subsetting matrix")
    if dl_wnd:
        selem = moph.disk(dl_wnd)
        sub = xr.apply_ufunc(
            cv2.dilate,
            A.chunk(dict(height=-1, width=-1)),
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            kwargs=dict(kernel=selem),
            dask="parallelized",
            output_dtypes=[A.dtype],
        )
        sub = sub > 0
    else:
        sub = xr.apply_ufunc(np.ones_like, A.compute())
    if update_background:
        A = xr.concat([A, b.assign_coords(unit_id=-1)], "unit_id")
        b_erd = xr.apply_ufunc(
            cv2.erode,
            b.chunk(dict(height=-1, width=-1)),
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            kwargs=dict(kernel=selem),
            dask="parallelized",
            output_dtypes=[b.dtype],
        )
        sub = xr.concat(
            [sub, (b_erd > 0).astype(bool).assign_coords(unit_id=-1)], "unit_id"
        )
        C = xr.concat([C, f.assign_coords(unit_id=-1)], "unit_id")
    sub = sub.persist()
    print("fitting spatial matrix")
    A_new = xr.apply_ufunc(
        update_spatial_perpx,
        Y.chunk(dict(frame=-1)),
        alpha,
        sub.chunk(dict(unit_id=-1)),
        C.chunk(dict(frame=-1, unit_id=-1)),
        input_core_dims=[["frame"], [], ["unit_id"], ["frame", "unit_id"]],
        output_core_dims=[["unit_id"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[Y.dtype],
    )
    try:
        with parallel_backend("dask"):
            A_new = A_new.persist()
    except ValueError:
        with da.config.set(scheduler=sched):
            A_new = A_new.persist()
    print("removing empty units")
    if zero_thres == "eps":
        zero_thres = np.finfo(A_new.dtype).eps
    A_new = A_new.where(A_new > zero_thres).fillna(0)
    non_empty = (A_new.sum(["width", "height"]) > 0).compute()
    A_new = A_new.where(non_empty, drop=True)
    C_new = C.where(non_empty, drop=True)
    A_new = rechunk_like(A_new, A).persist()
    C_new = rechunk_like(C_new, C).persist()
    if post_scal and len(A_new) > 0:
        print("post-hoc scaling")
        A_new_flt = A_new.stack(spatial=["height", "width"]).compute()
        Y_flt = Y.mean("frame").stack(spatial=["height", "width"]).compute()

        def lstsq(a, b):
            return np.linalg.lstsq(a, b, rcond=-1)[0]

        scale = xr.apply_ufunc(
            lstsq,
            A_new_flt,
            Y_flt,
            input_core_dims=[["spatial", "unit_id"], ["spatial"]],
            output_core_dims=[["unit_id"]],
        )
        C_mean = C.mean("frame").compute()
        scale = scale / C_mean
        A_new = A_new * scale
        try:
            A_new = A_new.persist()
        except np.linalg.LinAlgError:
            warnings.warn("post-hoc scaling failed", RuntimeWarning)
    if update_background:
        print("updating background")
        try:
            b_new = A_new.sel(unit_id=-1)
            b_new = b_new / da.array.linalg.norm(b_new.data)
            f_new = xr.apply_ufunc(
                da.array.tensordot,
                Y,
                b_new,
                input_core_dims=[["frame", "height", "width"], ["height", "width"]],
                output_core_dims=[["frame"]],
                kwargs=dict(axes=[(1, 2), (0, 1)]),
                dask="allowed",
            ).persist()
            A_new = A_new.drop(-1, "unit_id")
            C_new = C_new.drop(-1, "unit_id")
        except KeyError:
            print("background terms are empty")
            b_new = xr.zeros_like(b)
            f_new = xr.zeros_like(f)
    else:
        b_new = b
        f_new = f
    if normalize and len(A_new) > 0:
        print("normalizing result")
        A_norm = xr.apply_ufunc(
            darr.linalg.norm,
            A_new.stack(spatial=["height", "width"]),
            input_core_dims=[["spatial", "unit_id"]],
            output_core_dims=[["unit_id"]],
            kwargs=dict(axis=0, ord=1),
            dask="allowed",
        )
        A_new = (A_new / A_norm).persist()
    return A_new, b_new, C_new, f_new


def update_spatial_perpx(y, alpha, sub, C):
    res = np.zeros_like(sub, dtype=y.dtype)
    if np.sum(sub) > 0:
        C = C[:, sub]
        clf = LassoLars(alpha=alpha, positive=True)
        coef = clf.fit(C, y).coef_
        res[np.where(sub)[0]] = coef
    return res


def compute_trace(Y, A, b, C, f, noise_freq=None):
    nunits = len(A.coords["unit_id"])
    A_rechk = A.chunk(dict(height=-1, width=-1))
    C_rechk = C.chunk(dict(unit_id=-1))
    Y_rechk = Y.chunk(dict(height=-1, width=-1))
    AA = xr.apply_ufunc(
        da.array.tensordot,
        A_rechk,
        A_rechk.rename(dict(unit_id="unit_id_cp")),
        input_core_dims=[
            ["unit_id", "height", "width"],
            ["height", "width", "unit_id_cp"],
        ],
        output_core_dims=[["unit_id", "unit_id_cp"]],
        dask="allowed",
        kwargs=dict(axes=([1, 2], [0, 1])),
        output_dtypes=[A.dtype],
    )
    nA = (A_rechk ** 2).sum(["height", "width"]).compute()
    nA_inv = xr.apply_ufunc(
        lambda x: np.asarray(diags(x).todense()),
        1 / nA,
        input_core_dims=[["unit_id"]],
        output_core_dims=[["unit_id", "unit_id_cp"]],
        dask="parallelized",
        output_dtypes=[nA.dtype],
        output_sizes=dict(unit_id_cp=nunits),
    ).compute()
    nA_inv = nA_inv.assign_coords(unit_id_temp=AA.coords["unit_id_cp"])
    b = b.fillna(0).expand_dims("dot").chunk(dict(height=-1, width=-1))
    f = f.fillna(0).expand_dims("dot")
    B = xr.apply_ufunc(
        da.array.dot,
        b,
        f,
        input_core_dims=[["height", "width", "dot"], ["dot", "frame"]],
        output_core_dims=[["height", "width", "frame"]],
        dask="allowed",
        output_dtypes=[b.dtype],
    )
    Y = Y_rechk - B
    YA = xr.apply_ufunc(
        da.array.tensordot,
        Y,
        A_rechk,
        input_core_dims=[["frame", "height", "width"], ["height", "width", "unit_id"]],
        output_core_dims=[["frame", "unit_id"]],
        dask="allowed",
        kwargs=dict(axes=([1, 2], [0, 1])),
        output_dtypes=[A.dtype],
    ).rename(dict(unit_id="unit_id_cp"))
    YA_norm = xr.apply_ufunc(
        da.array.dot,
        YA,
        nA_inv,
        input_core_dims=[["frame", "unit_id_cp"], ["unit_id_cp", "unit_id"]],
        output_core_dims=[["frame", "unit_id"]],
        dask="allowed",
        output_dtypes=[YA.dtype],
    )
    CA = xr.apply_ufunc(
        da.array.dot,
        C_rechk,
        AA.chunk(dict(unit_id=-1, unit_id_cp=-1)),
        input_core_dims=[["frame", "unit_id"], ["unit_id", "unit_id_cp"]],
        output_core_dims=[["frame", "unit_id_cp"]],
        dask="allowed",
        output_dtypes=[C.dtype],
    )
    CA_norm = xr.apply_ufunc(
        da.array.dot,
        CA,
        nA_inv,
        input_core_dims=[["frame", "unit_id_cp"], ["unit_id_cp", "unit_id"]],
        output_core_dims=[["frame", "unit_id"]],
        dask="allowed",
        output_dtypes=[CA.dtype],
    )
    YrA = YA_norm - CA_norm + C_rechk
    if noise_freq:
        print("smoothing signals")
        but_b, but_a = butter(2, noise_freq, btype="low", analog=False)
        YrA_smth = xr.apply_ufunc(
            lambda x: lfilter(but_b, but_a, x),
            YrA.chunk(dict(frame=-1)),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[YrA.dtype],
        )
    else:
        YrA_smth = YrA
    return YrA_smth


def update_temporal(
    Y,
    A,
    b,
    C,
    f,
    sn_spatial,
    YrA=None,
    noise_freq=0.25,
    p=None,
    add_lag="p",
    jac_thres=0.1,
    use_spatial=False,
    sparse_penal=1,
    bseg=None,
    zero_thres=1e-8,
    max_iters=200,
    use_smooth=True,
    compute=True,
    normalize=False,
    post_scal=True,
    scs_fallback=False,
    sched="processes",
):
    print("grouping overlaping units")
    A_pos = (A > 0).astype(float)
    A_neg = (A == 0).astype(float)
    A_inter = xr.apply_ufunc(
        da.array.tensordot,
        A_pos,
        A_pos.rename(unit_id="unit_id_cp"),
        input_core_dims=[
            ["unit_id", "height", "width"],
            ["height", "width", "unit_id_cp"],
        ],
        output_core_dims=[["unit_id", "unit_id_cp"]],
        dask="allowed",
        kwargs=dict(axes=([1, 2], [0, 1])),
        output_dtypes=[A_pos.dtype],
    )
    A_union = xr.apply_ufunc(
        da.array.tensordot,
        A_neg,
        A_neg.rename(unit_id="unit_id_cp"),
        input_core_dims=[
            ["unit_id", "height", "width"],
            ["height", "width", "unit_id_cp"],
        ],
        output_core_dims=[["unit_id", "unit_id_cp"]],
        dask="allowed",
        kwargs=dict(axes=([1, 2], [0, 1])),
        output_dtypes=[A_neg.dtype],
    )
    A_jac = A_inter / (A.sizes["height"] * A.sizes["width"] - A_union)
    if compute:
        A_jac = A_jac.compute()
    unit_labels = xr.apply_ufunc(
        label_connected,
        A_jac > jac_thres,
        input_core_dims=[["unit_id", "unit_id_cp"]],
        kwargs=dict(only_connected=True),
        output_core_dims=[["unit_id"]],
    )
    if YrA is not None:
        YrA = YrA
    else:
        print("computing trace")
        YrA = compute_trace(Y, A, b, C, f).persist()
    YrA = YrA.chunk(dict(frame=-1, unit_id=1))
    YrA = YrA.assign_coords(unit_labels=unit_labels)
    if normalize:
        print("normalizing traces")
        YrA_norm = (YrA / YrA.sum("frame") * YrA.sizes["frame"]).persist()
    else:
        YrA_norm = YrA
    sn_temp = get_noise_fft(
        YrA_norm, noise_range=(noise_freq, 1), noise_method="sum"
    ).persist()
    sn_temp = sn_temp.assign_coords(unit_labels=unit_labels)
    if use_spatial:
        print("flattening spatial dimensions")
        Y_flt = Y.stack(spatial=("height", "width"))
        A_flt = A.stack(spatial=("height", "width")).assign_coords(
            unit_labels=unit_labels
        )
        sn_spatial = sn_spatial.stack(spatial=("height", "width"))
    if use_smooth:
        print("smoothing signals")
        but_b, but_a = butter(2, noise_freq, btype="low", analog=False)
        YrA_smth = xr.apply_ufunc(
            lambda x: lfilter(but_b, but_a, x),
            YrA_norm,
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[YrA.dtype],
        )
        if compute:
            YrA_smth = YrA_smth.persist()
            sn_temp_smth = get_noise_fft(
                YrA_smth, noise_range=(noise_freq, 1)
            ).persist()
            sn_temp_smth = sn_temp_smth.assign_coords(unit_labels=unit_labels)
    else:
        YrA_smth = YrA_norm
        sn_temp_smth = sn_temp
    if p is None:
        print("estimating order p for each neuron")
        p = xr.apply_ufunc(
            get_p,
            YrA_smth,
            input_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int],
        ).clip(1)
        if compute:
            p = p.compute()
        p_max = p.max().values
    else:
        p_max = p
    print("estimating AR coefficients")
    g = xr.apply_ufunc(
        get_ar_coef,
        YrA_smth.chunk(dict(frame=-1)),
        sn_temp_smth,
        p,
        input_core_dims=[["frame"], [], []],
        output_core_dims=[["lag"]],
        kwargs=dict(pad=p_max, add_lag=add_lag),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[sn_temp_smth.dtype],
        output_sizes=dict(lag=p_max),
    )
    g = g.assign_coords(lag=np.arange(1, p_max + 1), unit_labels=unit_labels)
    if compute:
        g = g.persist()
    print("updating isolated temporal components")
    if use_spatial == "full":
        result_iso = xr.apply_ufunc(
            update_temporal_cvxpy,
            Y_flt.chunk(dict(spatial=-1, frame=-1)),
            g.where(unit_labels == -1, drop=True).chunk(dict(lag=-1)),
            sn_spatial.chunk(dict(spatial=-1)),
            A_flt.where(unit_labels == -1, drop=True).chunk(dict(spatial=-1)),
            input_core_dims=[["spatial", "frame"], ["lag"], ["spatial"], ["spatial"]],
            output_core_dims=[["trace", "frame"]],
            vectorize=True,
            dask="parallelized",
            kwargs=dict(
                sparse_penal=sparse_penal,
                max_iters=max_iters,
                scs_fallback=scs_fallback,
            ),
            output_sizes=dict(trace=5),
            output_dtypes=[YrA.dtype],
        )
    else:
        gu_update = darr.gufunc(
            fct.partial(
                update_temporal_cvxpy,
                sparse_penal=sparse_penal,
                max_iters=max_iters,
                bseg=bseg,
                scs_fallback=scs_fallback,
            ),
            signature="(f),(l),()->(t,f)",
            vectorize=True,
            output_dtypes=[YrA.dtype],
            output_sizes=dict(t=5),
        )
        result_iso = xr.apply_ufunc(
            gu_update,
            YrA_norm.where(unit_labels == -1, drop=True).persist(),
            g.where(unit_labels == -1, drop=True).chunk(dict(lag=-1)).persist(),
            sn_temp.where(unit_labels == -1, drop=True).persist(),
            input_core_dims=[["frame"], ["lag"], []],
            output_core_dims=[["trace", "frame"]],
            dask="allowed",
        )
    if compute:
        with da.config.set(scheduler=sched):
            result_iso = result_iso.compute()
    print("updating overlapping temporal components")
    res_list = []
    g_ovlp = g.where(unit_labels >= 0, drop=True)
    if len(g_ovlp) > 0:
        for cur_labl, cur_g in g_ovlp.groupby("unit_labels"):
            if use_spatial:
                cur_A = A_flt_ovlp.where(unit_labels == cur_labl, drop=True)
                cur_res = delayed(xr.apply_ufunc)(
                    update_temporal_cvxpy,
                    Y_flt.chunk(dict(spatial=-1, frame=-1)),
                    cur_g.chunk(dict(lag=-1)),
                    sn_spatial.chunk(dict(spatial=-1)),
                    cur_A.chunk(dict(spatial=-1)),
                    input_core_dims=[
                        ["spatial", "frame"],
                        ["unit_id", "lag"],
                        ["spatial"],
                        ["unit_id", "spatial"],
                    ],
                    output_core_dims=[["trace", "unit_id", "frame"]],
                    dask="parallelized",
                    kwargs=dict(
                        sparse_penal=sparse_penal,
                        max_iters=max_iters,
                        scs_fallback=scs_fallback,
                    ),
                    output_sizes=dict(trace=5),
                    output_dtypes=[YrA.dtype],
                )
            else:
                cur_YrA = YrA_norm.where(unit_labels == cur_labl, drop=True)
                cur_sn_temp = sn_temp.where(unit_labels == cur_labl, drop=True)
                cur_res = delayed(xr.apply_ufunc)(
                    update_temporal_cvxpy,
                    cur_YrA.compute(),
                    cur_g.compute(),
                    cur_sn_temp.compute(),
                    input_core_dims=[
                        ["unit_id", "frame"],
                        ["unit_id", "lag"],
                        ["unit_id"],
                    ],
                    output_core_dims=[["trace", "unit_id", "frame"]],
                    dask="forbidden",
                    kwargs=dict(
                        sparse_penal=sparse_penal,
                        max_iters=max_iters,
                        bseg=bseg,
                        scs_fallback=scs_fallback,
                    ),
                    output_sizes=dict(trace=5),
                    output_dtypes=[YrA.dtype],
                )
                res_list.append(cur_res)
        if compute:
            with da.config.set(scheduler=sched):
                (result_ovlp,) = da.compute(res_list)
                result = (
                    xr.concat(result_ovlp + [result_iso], "unit_id")
                    .sortby("unit_id")
                    .drop("unit_labels")
                )
    else:
        result = result_iso.sortby("unit_id").drop("unit_labels")
    C_new = result.isel(trace=0).dropna("unit_id")
    S_new = result.isel(trace=1).dropna("unit_id")
    B_new = result.isel(trace=2).dropna("unit_id").squeeze()
    C0_new = result.isel(trace=3, frame=0).dropna("unit_id").squeeze()
    dc_new = result.isel(trace=4).dropna("unit_id")
    g_new = g.sel(unit_id=C_new.coords["unit_id"]).drop("unit_labels")
    if zero_thres:
        mask = S_new.where(S_new > zero_thres).fillna(0).sum("frame").astype(bool)
        mask_coord = mask.where(~mask, drop=True).coords["unit_id"].values
        print(
            "{} units dropped due to poor fit:\n {}".format(
                len(mask_coord), str(mask_coord)
            )
        )
    else:
        mask_coord = S_new.coords["unit_id"].values
        mask = xr.DataArray(
            np.ones(len(mask_coord)), dims=["unit_id"], coords=dict(unit_id=mask_coord)
        )
    C_new, S_new, C0_new, B_new, dc_new = (
        C_new.where(mask, drop=True),
        S_new.where(mask, drop=True),
        C0_new.where(mask, drop=True),
        B_new.where(mask, drop=True),
        dc_new.where(mask, drop=True),
    )
    YrA_new = YrA.drop("unit_labels").sel(unit_id=C_new.coords["unit_id"])
    sig_new = (C0_new * dc_new + B_new + C_new).persist()
    if post_scal and len(sig_new) > 0:
        print("post-hoc scaling")

        def lstsq(a, b):
            a = np.atleast_2d(a).T
            return np.linalg.lstsq(a, b, rcond=-1)[0]

        scal = xr.apply_ufunc(
            lstsq,
            sig_new.chunk(dict(frame=-1)),
            YrA_new.chunk(dict(frame=-1)),
            input_core_dims=[["frame"], ["frame"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[C_new.dtype],
        )
        scal = scal.persist()
        C_new = (C_new * scal).persist()
        S_new = (S_new * scal).persist()
        B_new = (B_new * scal).persist()
        C0_new = (C0_new * scal).persist()
        sig_new = (sig_new * scal).persist()
    else:
        scal = None
    if len(sig_new) > 0:
        C_new = rechunk_like(C_new.persist(), C)
        S_new = rechunk_like(S_new.persist(), C)
        B_new = rechunk_like(B_new.persist(), C)
        C0_new = rechunk_like(C0_new.persist(), C)
        g_new = rechunk_like(g_new.persist(), C)
        sig_new = rechunk_like(sig_new.persist(), C)
    return (YrA_norm, C_new, S_new, B_new, C0_new, sig_new, g_new, scal)


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


def update_temporal_cvxpy(y, g, sn, A=None, bseg=None, **kwargs):
    """
    spatial:
    (d, f), (u, p), (d), (d, u)
    (d, f), (p), (d), (d)
    trace:
    (u, f), (u, p), (u)
    (f), (p), ()
    """
    # get_parameters
    sparse_penal = kwargs.get("sparse_penal")
    max_iters = kwargs.get("max_iters")
    use_cons = kwargs.get("use_cons", False)
    scs = kwargs.get("scs_fallback")
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
        cur_c, cur_r = np.zeros(_T), np.zeros(_T)
        cur_c[0] = 1
        cur_r[0] = 1
        cur_c[1 : len(cur_g) + 1] = -cur_g
        # update G with toeplitz matrix
        G_ls.append(cvx.Constant(dia_matrix(toeplitz(cur_c, cur_r))))
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
    s = cvx.vstack([G_ls[u] * c[u, :] for u in range(_u)])  # spike train per unit
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
            _ = prob.solve(solver="ECOS", max_iters=max_iters)
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
                    "problem status is {}, returning null".format(prob.status),
                    RuntimeWarning,
                )
                return np.full((5, c.shape[0], c.shape[1]), np.nan).squeeze()
    if not (prob.status == "optimal"):
        warnings.warn("problem solved sub-optimally", RuntimeWarning)
    return np.stack(
        np.broadcast_arrays(
            c.value, s.value, b.value, c0.value.reshape((-1, 1)), dc_vec
        )
    ).squeeze()


def unit_merge(A, C, add_list=None, thres_corr=0.9):
    print("computing spatial overlap")
    A_bl = (A > 0).astype(np.float32).chunk(dict(unit_id="auto", height=-1, width=-1))
    A_ovlp = xr.apply_ufunc(
        da.array.tensordot,
        A_bl,
        A_bl.rename(unit_id="unit_id_cp"),
        input_core_dims=[
            ["unit_id", "height", "width"],
            ["height", "width", "unit_id_cp"],
        ],
        output_core_dims=[["unit_id", "unit_id_cp"]],
        dask="allowed",
        kwargs=dict(axes=([1, 2], [0, 1])),
        output_dtypes=[A_bl.dtype],
    )
    A_ovlp = A_ovlp.persist()
    print("computing temporal correlation")
    uid_idx = C.coords["unit_id"].values
    corr = xr.apply_ufunc(
        np.corrcoef,
        C.compute(),
        input_core_dims=[["unit_id", "frame"]],
        output_core_dims=[["unit_id", "unit_id_cp"]],
        output_sizes=dict(unit_id_cp=len(uid_idx)),
    )
    corr = corr.assign_coords(unit_id_cp=uid_idx)
    print("labeling units to be merged")
    adj = np.logical_and(A_ovlp > 0, corr > thres_corr)
    unit_labels = xr.apply_ufunc(
        label_connected,
        adj.compute(),
        input_core_dims=[["unit_id", "unit_id_cp"]],
        output_core_dims=[["unit_id"]],
    )
    print("merging units")
    A_merge = (
        A.assign_coords(unit_labels=unit_labels)
        .groupby("unit_labels")
        .sum("unit_id")
        .persist()
        .rename(unit_labels="unit_id")
    )
    C_merge = (
        C.assign_coords(unit_labels=unit_labels)
        .groupby("unit_labels")
        .mean("unit_id")
        .persist()
        .rename(unit_labels="unit_id")
    )
    A_merge = rechunk_like(A_merge, A)
    C_merge = rechunk_like(C_merge, C)
    if add_list:
        for ivar, var in enumerate(add_list):
            var_mrg = (
                var.assign_coords(unit_labels=unit_labels)
                .groupby("unit_labels")
                .mean("unit_id")
                .persist()
                .rename(unit_labels="unit_id")
            )
            add_list[ivar] = rechunk_like(var_mrg, var)
        return A_merge, C_merge, add_list
    else:
        return A_merge, C_merge


def label_connected(adj, only_connected=False):
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj)
    g = nx.convert_matrix.from_numpy_matrix(adj)
    labels = np.zeros(adj.shape[0], dtype=np.int)
    for icomp, comp in enumerate(nx.connected_components(g)):
        comp = list(comp)
        if only_connected and len(comp) == 1:
            labels[comp] = -1
        else:
            labels[comp] = icomp
    return labels


def smooth_sig(sig, freq, btype="low"):
    but_b, but_a = butter(2, freq, btype=btype, analog=False)
    sig_smth = xr.apply_ufunc(
        lambda x: lfilter(but_b, but_a, x),
        sig.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[sig.dtype],
    )
    return sig_smth
