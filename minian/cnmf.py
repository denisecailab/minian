import os
import warnings

import cv2
import cvxpy as cvx
import dask as da
import dask.array as darr
import networkx as nx
import numpy as np
import pyfftw.interfaces.numpy_fft as numpy_fft
import scipy.sparse
import sparse
import xarray as xr
import zarr
from scipy.linalg import lstsq, toeplitz
from scipy.ndimage import label
from scipy.signal import butter, lfilter, welch
from scipy.sparse import dia_matrix
from skimage import morphology as moph
from sklearn.linear_model import LassoLars
from sklearn.utils import parallel_backend
from statsmodels.tsa.stattools import acovf

from .utilities import rechunk_like, save_minian


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
    sn = xr.apply_ufunc(
        _noise_fft,
        varr,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        dask="parallelized",
        vectorize=True,
        kwargs=dict(noise_range=noise_range, noise_method=noise_method),
        output_dtypes=[np.float],
    )
    return sn


def _noise_fft(px, noise_range=(0.25, 0.5), noise_method="logmexp"):
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
    px_band = (1 / _T * np.abs(numpy_fft.rfft(px)) ** 2)[nr[0] : nr[1]]
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
    _T = Y.sizes["frame"]
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    if in_memory:
        C_store = C.compute().values
    else:
        C_path = os.path.join(intpath, C.name + ".zarr", C.name)
        C_store = zarr.open_array(C_path)
    print("estimating penalty parameter")
    alpha = sparse_penal * sn
    alpha = alpha.persist()
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
    sub = save_minian(
        sub.rename("sub"),
        intpath,
        overwrite=True,
        chunks={
            "unit_id": -1,
            "height": Y.data.chunksize[1],
            "width": Y.data.chunksize[2],
        },
    )
    sub.data = sub.data.map_blocks(sparse.COO)
    if update_background:
        b_in = rechunk_like(b > 0, Y).assign_coords(unit_id=-1).expand_dims("unit_id")
        b_in.data = b_in.data.map_blocks(sparse.COO)
        b_in = b_in.persist()
        sub = xr.concat([sub, b_in], "unit_id").chunk({"unit_id": -1})
        f_in = f.persist().data
    else:
        f_in = None
    print("fitting spatial matrix")
    A_new = update_spatial_block(
        Y.transpose("height", "width", "frame").data,
        alpha.data,
        sub.transpose("height", "width", "unit_id").data,
        C_store=C_store,
        f=f_in,
    )
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
    with parallel_backend("dask"):
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
            shape=(sub.shape[0], sub.shape[1], sub.shape[2]),
        )
    else:
        return sparse.zeros((sub.shape[0], sub.shape[1], sub.shape[2]))


def compute_trace(Y, A, b, C, f, noise_freq=None):
    fms = Y.coords["frame"]
    uid = A.coords["unit_id"]
    Y = Y.data
    A = A.data.map_blocks(sparse.COO).rechunk(-1).persist()
    C = C.data.map_blocks(sparse.COO).T
    b = (
        b.fillna(0)
        .data.map_blocks(sparse.COO)
        .reshape((1, Y.shape[1], Y.shape[2]))
        .rechunk(-1)
    )
    f = f.fillna(0).data.reshape((-1, 1))
    AtA = darr.tensordot(A, A, axes=[(1, 2), (1, 2)])
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
    YrA_ls = YrA.to_delayed().squeeze().tolist()
    for i in range(0, len(YrA_ls), 100):
        YrA_ls[i : i + 100] = da.compute(YrA_ls[i : i + 100])[0]
    YrA = darr.array(np.concatenate(YrA_ls, axis=0)).rechunk((-1, 1))
    YrA = xr.DataArray(
        YrA,
        dims=["frame", "unit_id"],
        coords={"frame": fms, "unit_id": uid},
    )
    if noise_freq:
        YrA = smooth_sig(YrA, noise_freq)
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
    if YrA is None:
        YrA = compute_trace(Y, A, b, C, f).persist()
    Ymask = (YrA > 0).any("frame").compute()
    A, C, YrA = A.sel(unit_id=Ymask), C.sel(unit_id=Ymask), YrA.sel(unit_id=Ymask)
    print("grouping overlaping units")
    A_sps = (A.data.map_blocks(sparse.COO) > 0).rechunk(-1).persist()
    A_inter = darr.tensordot(
        A_sps.astype(np.float32), A_sps.astype(np.float32), axes=[(1, 2), (1, 2)]
    ).compute()
    A_usum = np.tile(A_sps.sum(axis=(1, 2)).compute().todense(), (A_sps.shape[0], 1))
    A_usum = A_usum + A_usum.T
    jac = scipy.sparse.csc_matrix(A_inter / (A_usum - A_inter) > jac_thres)
    unit_labels = label_connected(jac)
    YrA = YrA.assign_coords(unit_labels=("unit_id", unit_labels))
    if normalize:
        print("normalizing traces")
        YrA_norm = (
            YrA.groupby("unit_labels").apply(lambda a: a / a.sum(axis=1).mean())
            * YrA.sizes["frame"]
        )
    else:
        YrA_norm = YrA
    tn = (
        get_noise_fft(YrA_norm, noise_range=(noise_freq, 1))
        .compute()
        .chunk({"unit_id": 1})
    )
    if use_smooth:
        print("smoothing traces")
        YrA_smth = smooth_sig(YrA_norm, noise_freq)
        tn_smth = get_noise_fft(YrA_smth, noise_range=(noise_freq, 1))
    else:
        YrA_smth = YrA
        tn_smth = tn
    if p is None:
        print("estimating order p for each neuron")
        p = xr.apply_ufunc(
            get_p,
            YrA_smth,
            input_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
        ).clip(1)
        p = p.compute()
        p_max = p.max().values
    else:
        p_max = p
    print("estimating AR coefficients")
    g = (
        xr.apply_ufunc(
            get_ar_coef,
            YrA_smth,
            tn_smth,
            p,
            input_core_dims=[["frame"], [], []],
            output_core_dims=[["lag"]],
            kwargs=dict(pad=p_max, add_lag=add_lag),
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            output_sizes=dict(lag=p_max),
        )
        .assign_coords(lag=np.arange(1, p_max + 1))
        .compute()
        .chunk({"unit_id": 1})
    )
    print("updating temporal components")
    res_ls = []
    uid_ls = []
    if concurrent_update:
        grp_dim = "unit_labels"
        C = C.assign_coords(unit_labels=("unit_id", unit_labels))
    else:
        grp_dim = "unit_id"
    if warm_start:
        C.data = C.data.map_blocks(scipy.sparse.csr_matrix)
    for cur_YrA, cur_g, cur_tn, cur_C in zip(
        YrA_norm.groupby(grp_dim),
        g.groupby(grp_dim),
        tn.groupby(grp_dim),
        C.groupby(grp_dim),
    ):
        uid_ls.append(cur_YrA[1].coords["unit_id"].values.reshape(-1))
        cur_YrA, cur_g, cur_tn, cur_C = (
            cur_YrA[1].data.rechunk(-1),
            cur_g[1].data.rechunk(-1),
            cur_tn[1].data.rechunk(-1),
            cur_C[1].data.rechunk(-1),
        )
        if not concurrent_update:
            cur_YrA, cur_g, cur_tn, cur_C = (
                cur_YrA.reshape((1, -1)),
                cur_g.reshape((1, -1)),
                cur_tn.reshape(1),
                cur_C.reshape((1, -1)),
            )
        if cur_YrA.shape[0] > 50:
            warnings.warn(
                "{} units are scheduled to update together, "
                "which might be too demanding. "
                "Consider merging the units "
                "or changing jaccard threshold".format(cur_YrA.shape[0])
            )
        if not warm_start:
            cur_C = None
        res = update_temporal_cvxpy(
            cur_YrA,
            cur_g,
            cur_tn,
            c_last=cur_C,
            bseg=bseg,
            sparse_penal=sparse_penal,
            max_iters=max_iters,
            scs_fallback=scs_fallback,
            zero_thres=zero_thres,
        )
        res_ls.append(res)
    uids_new = np.concatenate(uid_ls)
    res_ls = da.compute(res_ls)[0]
    C_new = darr.concatenate([darr.array(r[0]) for r in res_ls], axis=0)
    S_new = darr.concatenate([darr.array(r[1]) for r in res_ls], axis=0)
    b0_new = darr.concatenate([darr.array(r[2]) for r in res_ls], axis=0)
    c0_new = darr.concatenate([darr.array(r[3]) for r in res_ls], axis=0)
    mask = (S_new.sum(axis=1) > 0).compute()
    print("{} out of {} units dropped".format(mask.size - mask.nnz, len(Ymask)))
    mask = mask.todense()
    C_new, S_new, b0_new, c0_new, g, uids_new_ma = (
        C_new[mask, :],
        S_new[mask, :],
        b0_new[mask, :],
        c0_new[mask, :],
        g[mask, :],
        uids_new[mask],
    )
    sig_new = C_new + b0_new + c0_new
    YrA_new = YrA[mask, :]
    if post_scal and len(sig_new) > 0:
        print("post-hoc scaling")
        scal = (
            lstsq_vec(sig_new.rechunk((1, -1)), YrA_new.data).compute().reshape((-1, 1))
        )
        C_new, S_new, b0_new, c0_new = (
            C_new * scal,
            S_new * scal,
            b0_new * scal,
            c0_new * scal,
        )
    C_new = xr.DataArray(
        C_new.map_blocks(lambda a: a.todense(), dtype=float),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new_ma,
            "frame": YrA_new.coords["frame"],
        },
    )
    S_new = xr.DataArray(
        S_new.map_blocks(lambda a: a.todense(), dtype=float),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new_ma,
            "frame": YrA_new.coords["frame"].values,
        },
    )
    b0_new = xr.DataArray(
        b0_new.map_blocks(lambda a: a.todense(), dtype=float),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new_ma,
            "frame": YrA_new.coords["frame"].values,
        },
    )
    c0_new = xr.DataArray(
        c0_new.map_blocks(lambda a: a.todense(), dtype=float),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": uids_new_ma,
            "frame": YrA_new.coords["frame"].values,
        },
    )
    mask = xr.DataArray(
        mask, dims=["unit_id"], coords={"unit_id": uids_new}
    ).reindex_like(Ymask, fill_value=False)
    return C_new, S_new, b0_new, c0_new, g, mask


@darr.as_gufunc(signature="(f),(f)->()", output_dtypes=float)
def lstsq_vec(a, b):
    a = a.reshape((-1, 1)).todense()
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


@darr.as_gufunc(
    signature="(u,f),(u,p),(u)->(u,f),(u,f),(u,f),(u,f)",
    output_dtypes=(float, float, float, float),
)
def update_temporal_cvxpy(y, g, sn, A=None, bseg=None, **kwargs):
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
                return [sparse.zeros(c.shape, dtype=float)] * 4
    if not (prob.status == "optimal"):
        warnings.warn("problem solved sub-optimally", RuntimeWarning)
    c = np.where(c.value > zero_thres, c.value, 0)
    s = np.where(s.value > zero_thres, s.value, 0)
    b = np.where(b.value > zero_thres, b.value, 0)
    c0 = c0.value.reshape((-1, 1)) * dc_vec
    c0 = np.where(c0 > zero_thres, c0, 0)
    return sparse.COO(c), sparse.COO(s), sparse.COO(b), sparse.COO(c0)


def unit_merge(A, C, add_list=None, thres_corr=0.9):
    print("computing spatial overlap")
    A_sps = (A.data.map_blocks(sparse.COO) > 0).rechunk(-1).persist()
    A_inter = sparse.tril(
        darr.tensordot(
            A_sps.astype(np.float32), A_sps.astype(np.float32), axes=[(1, 2), (1, 2)]
        ).compute(),
        k=-1,
    )
    print("computing temporal correlation")
    corr_ls = []
    row_idx = []
    col_idx = []
    C = C - C.mean("frame")
    std = np.sqrt((C ** 2).sum("frame"))
    for i, j in zip(*A_inter.nonzero()):
        C_i, C_j, std_i, std_j = (
            C.isel(unit_id=i),
            C.isel(unit_id=j),
            std.isel(unit_id=i),
            std.isel(unit_id=j),
        )
        corr = (C_i * C_j).sum() / (std_i * std_j)
        corr_ls.append(corr)
        row_idx.append(i)
        col_idx.append(j)
    corr_ls = da.compute(corr_ls)[0]
    print("labeling units to be merged")
    adj = (
        scipy.sparse.csr_matrix((corr_ls, (row_idx, col_idx)), shape=A_inter.shape)
        > thres_corr
    )
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
    if method == "fft":
        _T = sig.sizes["frame"]
        if btype == "low":
            zero_range = slice(int(freq * _T), None)
        elif btype == "high":
            zero_range = slice(None, int(freq * _T))

        def filt_func(x):
            xfft = np.fft.rfft(x)
            xfft[zero_range] = 0
            return np.fft.irfft(xfft, len(x))

    elif method == "butter":
        but_b, but_a = butter(2, freq * 2, btype=btype, analog=False)
        filt_func = lambda x: lfilter(but_b, but_a, x)
    else:
        raise NotImplementedError(method)
    sig_smth = xr.apply_ufunc(
        filt_func,
        sig,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[sig.dtype],
    )
    return sig_smth


def compute_AtC(A, C):
    fm, h, w = (
        C.coords["frame"].values,
        A.coords["height"].values,
        A.coords["width"].values,
    )
    A = A.data.map_blocks(sparse.COO, dtype=A.dtype).rechunk(-1)
    C = C.transpose("frame", "unit_id").data.map_blocks(sparse.COO, dtype=C.dtype)
    AtC = darr.tensordot(C, A, axes=(1, 0)).map_blocks(
        lambda a: a.todense(), dtype=A.dtype
    )
    return xr.DataArray(
        AtC,
        dims=["frame", "height", "width"],
        coords={"frame": fm, "height": h, "width": w},
    )
