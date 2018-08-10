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
from sklearn.linear_model import LassoLars
from sklearn.externals.joblib import parallel_backend
from numba import jit, guvectorize
from skimage import morphology as moph
from timeit import timeit


def get_noise_fft(varr, noise_range=(0.25, 0.5), noise_method='logmexp'):
    _T = len(varr.coords['frame'])
    freq_crd = np.arange(0, 0.5 + 1 / _T, 1 / _T)
    print("computing fft of input")
    with ProgressBar():
        varr_fft = xr.apply_ufunc(
            dafft.rfft,
            varr,
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
    return A_new

@jit
def update_spatial_perpx(y, C, alpha, sub):
    res = np.zeros_like(sub, dtype=y.dtype)
    if np.sum(sub) > 0:
        C = C[:, sub]
        clf = LassoLars(alpha=alpha, positive=True)
        coef = clf.fit(C, y).coef_
        res[np.where(sub)[0]] = coef
    return res


@jit
def update_spatial_perpx_old(y, c, alpha):
    clf = LassoLars(alpha=alpha, positive=True)
    return clf.fit(c, y).coef_


@guvectorize(
    ['void(float64[:], float64[:, :], float64, float64[:])'],
    '(f),(f,u),()->(u)',
    target='cpu')
def update_spatial_perpx_vec(y, c, alpha, res):
    clf = LassoLars(alpha=alpha, positive=True)
    res[:] = clf.fit(c, y).coef_[:]


@da.array.as_gufunc(
    '(f),(f,u),()->(u)', output_dtypes=(np.float32), vectorize=True)
def update_spatial_perpx_da(y, c, alpha):
    import dask_ml.joblib
    from sklearn.externals.joblib import parallel_backend
    clf = LassoLars(alpha=alpha, positive=True)
    with parallel_backend('dask.distributed'):
        coef = clf.fit(c, y).coef_
    return coef


def update_spatial_chunk(Y, C, alpha):
    ht, wd, fm = Y.shape
    fm, ud = C.shape
    res = np.zeros((ht, wd, ud))
    for ih in range(ht):
        for iw in range(wd):
            cur_a = alpha[ih, iw]
            cur_y = Y[ih, iw, :]
            cur_clf = LassoLars(alpha=cur_a, positive=True)
            # with parallel_backend('threading'):
            coef = cur_clf.fit(C, np.ravel(cur_y)).coef_
            print("shapes:")
            print(C.shape, np.ravel(cur_y).shape)
            print("dtypes:")
            print(C.dtype, np.ravel(cur_y).dtype)
            print("avg time:")
            print(
                timeit(
                    lambda: cur_clf.fit(C, np.ravel(cur_y)).coef_,
                    number=2500))
    return res
