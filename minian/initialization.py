import numpy as np
import xarray as xr
import pandas as pd
import dask
import pyfftw.interfaces.numpy_fft as npfft
import dask.array.fft as dafft
import dask.array as da
import warnings
import cv2
from skimage.morphology import disk
from dask import delayed, compute
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.measurements import label
from scipy.stats import zscore, kstest
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert
from sklearn.mixture import GaussianMixture
from IPython.core.debugger import set_trace
from scipy.signal import butter, lfilter
from tqdm import tqdm_notebook
from .cnmf import smooth_sig, label_connected
from .utilities import get_optimal_chk, rechunk_like
from scipy.ndimage.filters import median_filter


def seeds_init(varr, wnd_size=500, method='rolling', stp_size=200, nchunk=100, max_wnd=10, diff_thres=2):
    print("constructing chunks")
    idx_fm = varr.coords['frame']
    nfm = len(idx_fm)
    if method == 'rolling':
        nstp = np.ceil(nfm / stp_size) + 1
        centers = np.linspace(0, nfm - 1, int(nstp))
        hwnd = np.ceil(wnd_size / 2)
        max_idx = list(
            map(lambda c: slice(int(np.floor(c - hwnd).clip(0)), int(np.ceil(c + hwnd))),
                centers))
    elif method == 'random':
        max_idx = [
            np.random.randint(0, nfm - 1, wnd_size) for _ in range(nchunk)
        ]
    res = []
    print("creating parallel scheme")
    res = [max_proj_frame(varr, cur_idx) for cur_idx in max_idx]
    max_res = xr.concat(res, 'sample').chunk(dict(sample=10))
    print("computing max projections")
    max_res = max_res.persist()
    print("calculating local maximum")
    loc_max = xr.apply_ufunc(
        local_max_roll,
        max_res.chunk(dict(height=-1, width=-1)),
        input_core_dims=[['height', 'width']],
        output_core_dims=[['height', 'width']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.uint8],
        kwargs=dict(k0=2, k1=max_wnd, diff=diff_thres)).sum('sample')
    loc_max = loc_max.compute()
    loc_max_flt = loc_max.stack(spatial=['height', 'width'])
    seeds = (loc_max_flt.where(loc_max_flt > 0, drop=True)
             .rename('seeds').to_dataframe().reset_index())
    return seeds[['height', 'width', 'seeds']].reset_index()


def max_proj_frame(varr, idx):
    return varr.isel(frame=idx).max('frame')

def local_max_roll(fm, k0, k1, diff):
    max_ls = []
    for ksize in range(k0, k1):
        selem = disk(ksize)
        fm_max = local_max(fm, selem, diff)
        max_ls.append(fm_max)
    lmax = (np.stack(max_ls, axis=0).sum(axis=0) > 0).astype(np.uint8)
    nlab, max_lab = cv2.connectedComponents(lmax)
    max_res = np.zeros_like(lmax)
    for lb in range(1, nlab):
        area = max_lab == lb
        if np.sum(area) > 1:
            crds = tuple(int(np.median(c)) for c in np.where(area))
            max_res[crds] = 1
        else:
            max_res[np.where(area)] = 1
    return max_res


def local_max(fm, k, diff=0):
    fm_max = cv2.dilate(fm, k)
    fm_min = cv2.erode(fm, k)
    fm_diff = ((fm_max - fm_min) > diff).astype(np.uint8)
    fm_max = (fm == fm_max).astype(np.uint8)
    return cv2.bitwise_and(fm_max, fm_diff).astype(np.uint8)


def gmm_refine(varr, seeds, q=(0.1, 99.9), n_components=2, valid_components=1, mean_mask=True):
    print("selecting seeds")
    varr_sub = varr.sel(
        spatial=[tuple(hw) for hw in seeds[['height', 'width']].values])
    print("computing peak-valley values")
    varr_valley = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[['frame']],
        kwargs=dict(q=q[0], axis=-1),
        dask='parallelized',
        output_dtypes=[varr_sub.dtype])
    varr_peak = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[['frame']],
        kwargs=dict(q=q[1], axis=-1),
        dask='parallelized',
        output_dtypes=[varr_sub.dtype])
    varr_pv = varr_peak - varr_valley
    varr_pv = varr_pv.compute()
    print("fitting GMM models")
    dat = varr_pv.values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(dat)
    idg = np.argsort(gmm.means_.reshape(-1))[-valid_components:]
    idx_valid = np.isin(gmm.predict(dat), idg)
    if mean_mask:
        idx_mean = dat > np.sort(gmm.means_)[0]
        idx_valid = np.logical_and(idx_mean.squeeze(), idx_valid)
    seeds['mask_gmm'] = idx_valid
    return seeds, varr_pv, gmm


def pnr_refine(varr, seeds, noise_freq=0.25, thres=1.5, q=(0.1, 99.9), med_wnd=None):
    print("selecting seeds")
    varr_sub = varr.sel(
        spatial=[tuple(hw) for hw in seeds[['height', 'width']].values])
    varr_sub = varr_sub.chunk(dict(frame=-1, spatial='auto'))
    if med_wnd:
        varr_base = xr.apply_ufunc(
            median_filter,
            varr_sub,
            input_core_dims=[['frame']],
            output_core_dims=[['frame']],
            dask='parallelized',
            kwargs=dict(size=med_wnd),
            vectorize=True,
            output_dtypes=[varr_sub.dtype])
        varr_sub = (varr_sub - varr_base).persist()
    print("computing peak-noise ratio")
    but_b, but_a = butter(2, noise_freq, btype='high', analog=False)
    varr_noise = xr.apply_ufunc(
        lambda x: lfilter(but_b, but_a, x),
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[['frame']],
        output_core_dims=[['frame']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[varr_sub.dtype])
    def ptp_q(x):
        return np.percentile(x, q[1]) - np.percentile(x, q[0])
    varr_sub_ptp = xr.apply_ufunc(
        ptp_q,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[['frame']],
        dask='parallelized',
        vectorize=True,
        output_dtypes=[varr_sub.dtype]).compute()
    varr_noise_ptp = xr.apply_ufunc(
        ptp_q,
        varr_noise.chunk(dict(frame=-1)).real,
        input_core_dims=[['frame']],
        dask='parallelized',
        vectorize=True,
        output_dtypes=[varr_sub.dtype]).compute()
    pnr = varr_sub_ptp / varr_noise_ptp
    if thres == 'auto':
        gmm = GaussianMixture(n_components=2)
        gmm.fit(np.nan_to_num(pnr.values.reshape(-1, 1)))
        idg = np.argsort(gmm.means_.reshape(-1))[-1]
        idx_valid = np.isin(gmm.predict(pnr.values.reshape(-1, 1)), idg)
        seeds['mask_pnr'] = idx_valid
    else:
        mask = pnr > thres
        mask = mask.compute()
        mask_df = mask.to_pandas().rename('mask_pnr').reset_index()
        seeds = pd.merge(seeds, mask_df, on=['height', 'width'], how='left')
        gmm = None
    return seeds, pnr, gmm


def intensity_refine(varr, seeds, thres_mul=2):
    try:
        fm_max = varr.max('frame')
    except ValueError:
        print("using input as max projection")
        fm_max = varr
    bins = np.around(
        fm_max.sizes['height'] * fm_max.sizes['width'] / 10).astype(int)
    hist, edges = np.histogram(fm_max, bins=bins)
    try:
        thres = edges[int(np.around(np.argmax(hist) * thres_mul))]
    except IndexError:
        print("threshold out of bound, returning input")
        return seeds
    mask = (fm_max > thres).stack(spatial=['height', 'width'])
    mask_df = mask.to_pandas().rename('mask_int').reset_index()
    seeds = pd.merge(seeds, mask_df, on=['height', 'width'], how='left')
    return seeds


def ks_refine(varr, seeds, sig=0.05):
    print("selecting seeds")
    varr_sub = varr.sel(
        spatial=[tuple(hw) for hw in seeds[['height', 'width']].values])
    print("performing KS test")
    ks = xr.apply_ufunc(
        lambda x: kstest(zscore(x), 'norm')[1],
        varr_sub.chunk(dict(frame=-1, spatial='auto')),
        input_core_dims=[['frame']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float])
    mask = ks < sig
    mask_df = mask.to_pandas().rename('mask_ks').reset_index()
    seeds = pd.merge(seeds, mask_df, on=['height', 'width'], how='left')
    return seeds


def seeds_merge(varr, seeds, thres_dist=5, thres_corr=0.6, noise_freq='envelope'):
    crds = [tuple([h, w]) for h, w in seeds[['height', 'width']].values]
    nsmp = len(crds)
    varr_sub = varr.sel(spatial=crds)
    varr_max = varr_sub.max('frame').compute()
    print("computing distance")
    dist = xr.DataArray(
        squareform(pdist(seeds[['height', 'width']].values)),
        dims=['sampleA', 'sampleB'],
        coords=dict(
            sampleA=np.arange(nsmp),
            sampleB=np.arange(nsmp)))
    if noise_freq:
        if noise_freq == 'envelope':
            print("computing hilbert transform")
            varr_sub = xr.apply_ufunc(
                lambda x: abs(hilbert(x)),
                varr_sub.chunk(dict(frame=-1, spatial='auto')),
                input_core_dims=[['frame']],
                output_core_dims=[['frame']],
                vectorize=True,
                output_dtypes=[varr_sub.dtype],
                dask='parallelized')
        else:
            varr_sub = smooth_sig(varr_sub, noise_freq)
    corr = (xr.apply_ufunc(
        da.corrcoef,
        varr_sub.chunk(dict(spatial='auto', frame=-1)),
        input_core_dims=[['spatial', 'frame']],
        output_core_dims=[['sampleA', 'sampleB']],
        dask='allowed',
        output_sizes=dict(sampleA=nsmp, sampleB=nsmp),
        output_dtypes=[float])
            .assign_coords(
                sampleA=np.arange(nsmp),
                sampleB=np.arange(nsmp)))
    print("computing correlations")
    corr = corr.compute()
    adj = np.logical_and(dist < thres_dist, corr > thres_corr)
    adj = adj.compute()
    np.fill_diagonal(adj.values, 0)
    iso = adj.sum('sampleB')
    iso = iso.where(iso == 0).dropna('sampleA')
    labels = label_connected(adj.values)
    uids = adj.coords['sampleA'].values
    seeds_final = set(iso.coords['sampleA'].data.tolist())
    for cur_cmp in np.unique(labels):
        cur_smp = uids[np.where(labels == cur_cmp)[0]]
        cur_max = varr_max.isel(spatial=cur_smp)
        max_seed = cur_smp[np.argmax(cur_max.data)]
        seeds_final.add(max_seed)
    seeds['mask_mrg'] = False
    seeds.loc[list(seeds_final), 'mask_mrg'] = True
    return seeds


def initialize(varr, seeds, thres_corr=0.8, wnd=10, noise_freq=None):
    print("creating parallel schedule")
    harr, warr = seeds['height'].values, seeds['width'].values
    varr_rechk = varr.chunk(dict(frame=-1))
    res_ls = [init_perseed(varr_rechk, h, w, wnd, thres_corr, noise_freq)
              for h, w in zip(harr, warr)]
    print("computing ROIs")
    res_ls = dask.compute(res_ls)[0]
    print("concatenating results")
    A = (xr.concat([r[0] for r in res_ls], 'unit_id')
         .assign_coords(unit_id = np.arange(len(res_ls))))
    C = (xr.concat([r[1] for r in res_ls], 'unit_id')
         .assign_coords(unit_id = np.arange(len(res_ls))))
    print("initializing backgrounds")
    A = A.reindex_like(varr.isel(frame=0)).fillna(0)
    chk = {d: c for d, c in zip(varr.dims, varr.chunks)}
    uchkA = get_optimal_chk(varr, A)['unit_id']
    uchkC = get_optimal_chk(varr, C)['unit_id']
    uchk = min(uchkA, uchkC)
    A = A.chunk(dict(height=chk['height'], width=chk['width'], unit_id=uchk))
    C = C.chunk(dict(frame=chk['frame'], unit_id=uchk))
    A_mask = A.sum('unit_id') == 0
    Yb = varr.where(A_mask, 0)
    b = Yb.mean('frame').persist()
    f = Yb.mean(['height', 'width']).persist()
    b = rechunk_like(b, varr)
    return A, C, b, f


def init_perseed(varr, h, w, wnd, thres_corr, noise_freq):
    ih = np.where(varr.coords['height'] == h)[0][0]
    iw = np.where(varr.coords['width'] == w)[0][0]
    h_sur, w_sur = (slice(max(ih - wnd, 0), ih + wnd),
                    slice(max(iw - wnd, 0), iw + wnd))
    sur = varr.isel(height=h_sur, width=w_sur)
    sur_flt = sur.stack(spatial=['height', 'width'])
    ih = np.where(sur.coords['height'] == h)[0][0]
    iw = np.where(sur.coords['width'] == w)[0][0]
    sp_idxs = sur_flt.coords['spatial'].values
    if noise_freq:
        sur_smth = smooth_sig(sur_flt, noise_freq)
    else:
        sur_smth = sur_flt
    corr = xr.apply_ufunc(
        da.corrcoef,
        sur_smth,
        input_core_dims=[['spatial', 'frame']],
        output_core_dims=[['spatial', 'spatial_cp']],
        dask='allowed',
        output_sizes=dict(spatial_cp=len(sp_idxs)))
    sd_id = np.ravel_multi_index(
        (ih, iw),
        (sur.sizes['height'], sur.sizes['width']))
    corr = (corr.isel(spatial_cp=sd_id)
            .squeeze().unstack('spatial'))
    mask = corr > thres_corr
    mask_lb = xr.apply_ufunc(da_label, mask, dask='allowed')
    sd_lb = mask_lb.isel(height=ih, width=iw)
    mask = (mask_lb == sd_lb)
    sur = sur.where(mask, 0)
    corr = corr.where(mask, 0)
    corr_norm = corr / corr.sum()
    C = xr.apply_ufunc(
        da.tensordot, sur, corr_norm,
        input_core_dims=[['frame', 'height', 'width'], ['height', 'width']],
        output_core_dims=[['frame']],
        kwargs=dict(axes=[(1, 2), (0, 1)]),
        dask='allowed')
    return corr, C


@da.as_gufunc(signature="(h, w)->(h, w)", output_dtypes=int, allow_rechunk=True)
def da_label(im):
    return label(im)[0]
