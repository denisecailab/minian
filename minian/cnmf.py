import numpy as np
import xarray as xr
import pandas as pd
import dask as da
import graph_tool.all as gt
import numba as nb
import dask.array.fft as dafft
import dask_ml.joblib
import graph_tool.all as gt
from dask import delayed, compute
from dask.diagnostics import ProgressBar, Profiler
from dask.distributed import progress
from IPython.core.debugger import set_trace
from scipy.ndimage import gaussian_filter, label
from scipy.signal import welch, butter, lfilter
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


def noise_welch(y, noise_range, noise_method):
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


def update_spatial(Y,
                   A,
                   b,
                   C,
                   f,
                   sn,
                   gs_sigma=6,
                   dl_wnd=5,
                   sparse_penal=0.5,
                   update_background=False,
                   compute=True):
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
        alpha = sparse_penal * sn * np.sqrt(np.max(np.diag(cct))) / _T
    print("computing subsetting matrix")
    if update_background:
        A = xr.concat([A, b.assign_coords(unit_id=-1)], 'unit_id')
        C = xr.concat([C, f.assign_coords(unit_id=-1)], 'unit_id')
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
    if update_background:
        b_new = A_new.sel(unit_id=-1)
        A_new = A_new.drop(-1, 'unit_id')
        C_new = C.drop(-1, 'unit_id')
    else:
        b_new = b
    print("removing empty units")
    non_empty = A_new.sum(['width', 'height']) > 0
    A_new = A_new.where(non_empty, drop=True)
    C_new = C.where(non_empty, drop=True)
    if compute:
        with ProgressBar():
            A_new, C_new = A_new.compute(), C_new.compute()
    return A_new, b_new, C_new, f


def update_spatial_perpx(y, C, alpha, sub):
    res = np.zeros_like(sub, dtype=y.dtype)
    if np.sum(sub) > 0:
        C = C[:, sub]
        clf = LassoLars(alpha=alpha, positive=True)
        coef = clf.fit(C, y).coef_
        res[np.where(sub)[0]] = coef
    return res


def update_temporal(Y,
                    A,
                    b,
                    C,
                    f,
                    sn,
                    noise_freq=0.25,
                    p=None,
                    add_lag='p',
                    sparse_penal=1,
                    max_iters=500,
                    use_smooth=True,
                    compute=True):
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
    sn_temp = get_noise_welch(YrA, noise_range=(noise_freq, 1))
    if use_smooth:
        print("smoothing signals")
        but_b, but_a = butter(2, noise_freq, btype='low', analog=False)
        YrA_smth = xr.apply_ufunc(
            lambda x: lfilter(but_b, but_a, x),
            YrA.chunk(dict(frame=-1)),
            input_core_dims=[['frame']],
            output_core_dims=[['frame']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[YrA.dtype])
        if compute:
            with ProgressBar():
                YrA_smth = YrA_smth.compute()
                sn_temp_smth = get_noise_welch(
                    YrA_smth, noise_range=(noise_freq, 1))
    else:
        YrA_smth = YrA
        sn_temp_smth = sn_temp
    if p is None:
        print("estimating order p for each neuron")
        p = xr.apply_ufunc(
            get_p,
            YrA_smth.chunk(dict(frame=-1)),
            input_core_dims=[['frame']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.int]).clip(1)
        if compute:
            with ProgressBar():
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
        input_core_dims=[['frame'], [], []],
        output_core_dims=[['lag']],
        kwargs=dict(pad=p_max, add_lag=add_lag),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[sn.dtype],
        output_sizes=dict(lag=p_max))
    g = g.assign_coords(lag=np.arange(1, p_max + 1))
    if compute:
        with ProgressBar():
            g = g.compute()
    print("updating temporal components")
    C_new, S_new, B_new, C0_new = xr.apply_ufunc(
        update_temporal_cvxpy,
        YrA.compute(),
        g.compute(),
        sn_temp.compute(),
        input_core_dims=[['frame'], ['lag'], []],
        output_core_dims=[['frame'], ['frame'], [], []],
        vectorize=True,
        dask='forbidden',
        kwargs=dict(sparse_penal=sparse_penal, max_iters=max_iters),
        output_dtypes=[YrA.dtype, YrA.dtype, YrA.dtype, YrA.dtype])
    if compute:
        with ProgressBar():
            C_new, S_new, B_new, C0_new = C_new.compute(), S_new.compute(
            ), B_new.compute(), C0_new.compute()
    return YrA, C_new, S_new, B_new, C0_new, g


def get_ar_coef(y, sn, p, add_lag, pad=None):
    if add_lag is 'p':
        max_lag = p * 2
    else:
        max_lag = p + add_lag
    cov = acovf(y, fft=True)
    C_mat = toeplitz(cov[:max_lag], cov[:p]) - sn**2 * np.eye(max_lag, p)
    g = lstsq(C_mat, cov[1:max_lag + 1])[0]
    if pad:
        res = np.zeros(pad)
        res[:len(g)] = g
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


def update_temporal_cvxpy(y, g, sn, sparse_penal, max_iters, A=None):
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
        cur_c[1:len(cur_g) + 1] = -cur_g
        # update G with toeplitz matrix
        G[cur_u, :, :] = toeplitz(cur_c, cur_r)
        # update dc_vec
        cur_gr = np.roots(cur_c)
        dc_vec[cur_u, :] = np.max(cur_gr.real)**np.arange(_T)
    # get noise threshold
    thres_sn = sn * np.sqrt(_T)
    # construct variables
    b = cvx.Variable(_u)  # baseline fluorescence per unit
    c0 = cvx.Variable(_u)  # initial fluorescence per unit
    c = cvx.Variable((_u, _T))  # calcium trace per unit
    s = cvx.vstack(
        [G[u, :, :] * c[u, :] for u in range(_u)])  # spike train per unit
    # residual noise per unit
    if A is not None:
        noise = cvx.vstack([
            y[px, :] - (A * c)[px, :] - (A * b)[px, :] -
            (A * cvx.diag(c0) * dc_vec)[px, :] for ps in range(_d)
        ])
    else:
        noise = cvx.vstack([
            y[u, :] - c[u, :] - b[u] - c0[u] * dc_vec[u, :] for u in range(_u)
        ])
    noise = cvx.vstack(
        [cvx.norm(noise[i, :], 2) for i in range(noise.shape[0])])
    # construct constraints
    cons = []
    cons.append(b >= np.min(y, axis=-1))  # baseline larger than minimum
    cons.append(c0 >= 0)  # initial fluorescence larger than 0
    cons.append(s >= 0)  # spike train non-negativity
    # noise constraints
    cons_noise = [noise[i] <= thres_sn[i] for i in range(thres_sn.shape[0])]
    # objective
    # try:
    #     obj = cvx.Minimize(cvx.norm(s, 1))
    #     prob = cvx.Problem(obj, cons + cons_noise)
    #     res = prob.solve(solver='ECOS', max_iters=max_iters)
    #     if not (prob.status == 'optimal'
    #             or prob.status == 'optimal_inaccurate'):
    #         print("constrained version of problem infeasible")
    #         raise ValueError
    # except (ValueError, cvx.SolverError):
    # Pr = np.zeros(_T)
    # Pr[1:10] = 1
    # P = toeplitz(Pr)
    # sPs = cvx.vstack([s * P * s.T for u in range(s.shape[0])])
    lam = sn * sparse_penal
    obj = cvx.Minimize(cvx.sum(noise) + lam * cvx.norm(s, 1, axis=1))
    prob = cvx.Problem(obj, cons)
    try:
        res = prob.solve(solver='ECOS', max_iters=max_iters)
        if not prob.status is 'optimal':
            print(
                "problem solved sub-optimally or unfeasible, trying SCS solver"
            )
            raise ValueError
    except (ValueError, cvx.SolverError):
        res = prob.solve(solver='SCS', verbose=True)
    return c.value, s.value, b.value, c0.value


def unit_merge(A, C, thres_corr=0.9):
    print("computing spatial overlap")
    A_ovlp = A.dot(A.rename(unit_id='unit_id_cp'), ['height', 'width'])
    uid_idx = C.coords['unit_id'].values
    print("computing temporal correlation")
    corr = xr.apply_ufunc(
        np.corrcoef,
        C.compute(),
        input_core_dims=[['unit_id', 'frame']],
        output_core_dims=[['unit_id', 'unit_id_cp']],
        output_sizes=dict(unit_id_cp=len(uid_idx)))
    corr = corr.assign_coords(unit_id_cp=uid_idx)
    print("labeling units to be merged")
    adj = np.logical_and(A_ovlp > 0, corr > thres_corr)
    unit_labels = xr.apply_ufunc(
        label_connected,
        adj.compute(),
        input_core_dims=[['unit_id', 'unit_id_cp']],
        output_core_dims=[['unit_id']],
        output_dtypes=[np.int32])
    print("merging units")
    A_merge = A.assign_coords(unit_labels=unit_labels).groupby(
        'unit_labels').sum('unit_id').rename(unit_labels='unit_id')
    C_merge = C.assign_coords(unit_labels=unit_labels).groupby(
        'unit_labels').mean('unit_id').rename(unit_labels='unit_id')
    return A_merge, C_merge


def label_connected(adj, only_connected=False):
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj)
    eg_list = []
    for idx, wt in np.ndenumerate(adj):
        if wt > 0:
            eg_list.append(list(idx))
    g = gt.Graph(directed=False)
    vmap = g.add_vertex(adj.shape[0])
    gmap = g.add_edge_list(eg_list)
    comp, hist = gt.label_components(g)
    labels = np.array(comp.a)
    if only_connected:
        labels[np.isin(labels, np.where(hist == 1)[0])] = -1
    return labels
