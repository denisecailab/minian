import numpy as np
import xarray as xr
import pandas as pd
import dask as da
import graph_tool.all as gt
import numba as nb
import dask.array.fft as dafft
import dask_ml.joblib
from dask import delayed, compute
from dask.diagnostics import ProgressBar, Profiler
from dask.distributed import progress
from IPython.core.debugger import set_trace
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from scipy.sparse import diags
from scipy.linalg import toeplitz, lstsq
from sklearn.linear_model import LassoLars
from sklearn.externals.joblib import parallel_backend
from numba import jit, guvectorize
from skimage import morphology as moph
from statsmodels.tsa.stattools import acovf
import cvxpy as cvx
from timeit import timeit


def get_noise_fft(varr, noise_range=(0.25, 0.5), noise_method='logmexp'):
    _T = len(varr.coords['frame'])
    freq_crd = np.arange(0, 0.5 + 1 / _T, 1 / _T)
    print("computing fft of input")
    with ProgressBar():
        varr_fft = xr.apply_ufunc(
            dafft.rfft,
            varr.chunk(dict(frame=-1)),
            input_core_dims=[['frame']],
            output_core_dims=[['freq']],
            dask='allowed',
            output_sizes=dict(freq=len(freq_crd)),
            output_dtypes=[np.complex_]).persist()
    print("computing power of noise")
    varr_fft = varr_fft.assign_coords(freq=freq_crd)
    varr_band = varr_fft.sel(freq=slice(*noise_range))
    varr_psd = 1 / _T * np.abs(varr_band)**2
    with ProgressBar():
        varr_psd = varr_psd.persist()
    print("estimating noise using method {}".format(noise_method))
    if noise_method == 'mean':
        sn = np.sqrt(varr_psd.mean('freq'))
    elif noise_method == 'median':
        sn = np.sqrt(varr_psd.median('freq'))
    elif noise_method == 'logmexp':
        eps = np.finfo(varr_psd.dtype).eps
        sn = np.sqrt(np.exp(np.log(varr_psd + eps).mean('freq')))
    with ProgressBar():
        sn = sn.persist()
    return sn, varr_psd


def get_noise_welch(varr,
                    noise_range=(0.25, 0.5),
                    noise_method='logmexp',
                    compute=True):
    print("estimating noise")
    sn = xr.apply_ufunc(
        noise_welch,
        varr.chunk(dict(frame=-1)),
        input_core_dims=[['frame']],
        dask='parallelized',
        vectorize=True,
        kwargs=dict(noise_range=noise_range, noise_method=noise_method),
        output_dtypes=[varr.dtype])
    if compute:
        with ProgressBar():
            sn = sn.compute()
    return sn


def noise_welch(y, noise_range=(0.25, 0.5), noise_method='logmexp'):
    ff, Pxx = welch(y)
    mask0, mask1 = ff > noise_range[0], ff < noise_range[1]
    mask = np.logical_and(mask0, mask1)
    Pxx_ind = Pxx[mask]
    sn = {
        'mean': lambda x: np.sqrt(np.mean(x / 2)),
        'median': lambda x: np.sqrt(np.median(x / 2)),
        'logmexp': lambda x: np.sqrt(np.exp(np.mean(np.log(x / 2))))
    }[noise_method](Pxx_ind)
    return sn


def update_spatial(Y, A, b, C, f, sn, gs_sigma=6, dl_wnd=5, compute=True):
    _T = len(Y.coords['frame'])
    # print(
    #     "gaussian filtering on spatial matrix with sigma: {}".format(gs_sigma))
    # A_gs = xr.apply_ufunc(
    #         gaussian_filter,
    #         A,
    #         input_core_dims=[['height', 'width']],
    #         output_core_dims=[['height', 'width']],
    #         vectorize=True,
    #         kwargs=dict(sigma=gs_sigma),
    #         dask='parallelized',
    #         output_dtypes=[A.dtype])
    # if compute:
    #     with ProgressBar():
    #         A_gs = A_gs.persist()
    print("estimating penalty parameter")
    with ProgressBar():
        cct = C.dot(C, 'frame')
        alpha = .5 * sn * np.sqrt(np.max(np.diag(cct))) / _T
    print("computing subsetting matrix")
    if dl_wnd:
        selem = moph.disk(dl_wnd)
        sub = xr.apply_ufunc(
            moph.dilation,
            A.fillna(0).chunk(dict(height=-1, width=-1)),
            input_core_dims=[['height', 'width']],
            output_core_dims=[['height', 'width']],
            vectorize=True,
            kwargs=dict(selem=selem),
            dask='parallelized',
            output_dtypes=[A.dtype])
        sub = sub > 0
    else:
        sub = xr.apply_ufunc(np.ones_like, A.compute())
    if compute:
        with ProgressBar():
            sub = sub.persist().astype(np.bool)
    print("fitting spatial matrix")
    A_new = xr.apply_ufunc(
        update_spatial_perpx,
        Y.chunk(dict(frame=-1)),
        C.chunk(dict(frame=-1, unit_id=-1)),
        alpha,
        sub.chunk(dict(unit_id=-1)),
        input_core_dims=[['frame'], ['frame', 'unit_id'], [], ['unit_id']],
        output_core_dims=[['unit_id']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[Y.dtype],
    )
    if compute:
        with ProgressBar():
            A_new = A_new.compute()
    print("removing empty units")
    non_empty = A_new.sum(['width', 'height']) > 0
    A_new = A_new.where(non_empty, drop=True)
    C_new = C.where(non_empty, drop=True)
    return A_new, C_new


@jit
def update_spatial_perpx(y, C, alpha, sub):
    res = np.zeros_like(sub, dtype=y.dtype)
    if np.sum(sub) > 0:
        C = C[:, sub]
        clf = LassoLars(alpha=alpha, positive=True)
        coef = clf.fit(C, y).coef_
        res[np.where(sub)[0]] = coef
    return res


def update_temporal(Y, A, b, C, f, sn, p=2, max_lag=50, compute=True):
    print("computing trace")
    YA = Y.dot(A, ['height', 'width']).rename(dict(unit_id='unit_id_temp'))
    AA = A.dot(A.rename(dict(unit_id='unit_id_temp')), ['height', 'width'])
    nA = (A**2).sum(['height', 'width'])
    nA_inv = xr.apply_ufunc(
        lambda x: np.asarray(diags(x).todense()),
        1 / nA,
        input_core_dims=[['unit_id']],
        output_core_dims=[['unit_id', 'unit_id_temp']],
        dask='parallelized',
        output_dtypes=[nA.dtype])
    nA_inv = nA_inv.assign_coords(unit_id_temp=AA.coords['unit_id_temp'])
    YrA = YA.dot(nA_inv, 'unit_id_temp') - C.dot(AA, 'unit_id').dot(
        nA_inv, 'unit_id_temp')
    YrA = YrA + C
    if compute:
        with ProgressBar():
            YrA = YrA.compute()
    sn = get_noise_welch(YrA)
    print("estimating AR coefficients")
    g = xr.apply_ufunc(
        get_ar_coef,
        YrA.chunk(dict(frame=-1)),
        sn,
        input_core_dims=[['frame'], []],
        output_core_dims=[['lag']],
        kwargs=dict(p=p, max_lag=max_lag),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[sn.dtype],
        output_sizes=dict(lag=p))
    g = g.assign_coords(lag=np.arange(1, p + 1))
    if compute:
        with ProgressBar():
            g = g.compute()
    return YrA, sn, g


def get_ar_coef(y, sn, p, max_lag):
    cov = acovf(y, fft=True)
    C_mat = toeplitz(cov[:max_lag], cov[:p - 1]) - sn**2 * np.eye(max_lag, p)
    g = lstsq(C_mat, cov[1:max_lag + 1])[0]
    return g


def update_temporal_cvxpy(y, g, sn, A=None):
    """
    spatial:
    (d, f), (u, p), (d), (d, u)
    (d, f), (p), (d), (d)
    trace:
    (u, f), (u, p), (u)
    (f), (p), ()
    """
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
    G = np.zeros((_u, _T, _T))
    dc_vec = np.zeros((_u, _T))
    for cur_u in range(_u):
        cur_g = g[cur_u, :]
        # construct first column and row
        cur_c, cur_r = np.zeros(_T), np.zeros(_T)
        cur_c[0] = 1
        cur_r[0] = 1
        try:
            cur_c[1:len(cur_g) + 1] = -cur_g
        except:
            set_trace()
        # update G with toeplitz matrix
        G[cur_u, :, :] = toeplitz(cur_c, cur_r)
        # update dc_vec
        cur_gr = np.roots(cur_c)
        dc_vec[cur_u, :] = np.max(cur_gr)**np.arange(_T)
    # get noise threshold
    thres_sn = sn * np.sqrt(_T)
    # construct variables
    b = cvx.Variable(_u)  # baseline fluorescence per unit
    c0 = cvx.Variable(_u)  # initial fluorescence per unit
    c = cvx.Variable((_u, _T))  # calcium trace per unit
    # construct constraints
    cons = []
    cons.append(b >= np.min(y, axis=-1))  # baseline larger than minimum
    cons.append(c0 >= 0)  # initial fluorescence larger than 0
    # non-negative constraints
    for cur_u in range(_u):
        cons.append(G[cur_u, :, :] * c[cur_u, :] >= 0)
    # noise constraints
    if A is not None:
        noise = cvx.vstack([
            y[px, :] - A * c[px, :] - A * b[px, :] -
            A * cvx.diag(c0) * dc_vec[px, :] for ps in range(_d)
        ])
    else:
        noise = cvx.vstack([
            y[u, :] - c[u, :] - b[u] - c0[u] * dc_vec[u, :] for u in range(_u)
        ])
    cons_noise = [cvx.norm(noise, 2, axis=1) <= thres_sn]
    # objective
    s = cvx.vstack([G[u, :, :] * c[u, :] for u in range(_u)])
    obj = cvx.Minimize(cvx.norm(s, 1))
    prob = cvx.Problem(obj, cons + cons_noise)
    res = prob.solve(solver='ECOS' verbose=True)
    # lam = sn / 500
    # obj = cvx.minimize(cvx.norm(noise, 2) + lam * cvx.norm(s, 1, axis=1))
    # prob = cvx.problem(obj, cons)
    # res = prob.solve(solver='SCS', verbose=True)
    return prob, res, c, s, b, c0, noise
