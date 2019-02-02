import numpy as np
import xarray as xr
import pandas as pd
import dask
import pyfftw.interfaces.numpy_fft as npfft
import dask.array.fft as dafft
import dask.array as da
import warnings
from dask import delayed, compute
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
from scipy.stats import zscore, kstest
from scipy.spatial.distance import pdist, squareform
from scipy.signal import hilbert
from sklearn.mixture import GaussianMixture
from IPython.core.debugger import set_trace
from scipy.signal import butter, lfilter
from tqdm import tqdm_notebook
from .cnmf import smooth_sig, label_connected
from scipy.ndimage.filters import median_filter


def seeds_init(varr, wnd_size=500, method='rolling', stp_size=200, nchunk=100, max_wnd=10):
    print("constructing chunks")
    idx_fm = varr.coords['frame']
    nfm = len(idx_fm)
    if method == 'rolling':
        nstp = np.ceil(nfm / stp_size)
        centers = np.linspace(0, nfm - 1, nstp)
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
    print("computing max projection")
    max_res = max_res.persist()
    print("calculating local maximum")
    loc_max = xr.apply_ufunc(
        local_max,
        max_res,
        input_core_dims=[['height', 'width']],
        output_core_dims=[['height', 'width']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.uint8],
        kwargs=dict(wnd=max_wnd)).sum('sample')
    loc_max = loc_max.compute()
    loc_max_flt = loc_max.stack(spatial=['height', 'width'])
    seeds = (loc_max_flt.where(loc_max_flt > 0, drop=True)
             .rename('seeds').to_dataframe().reset_index())
    return seeds[['height', 'width', 'seeds']].reset_index()


def max_proj_frame(varr, idx):
    return varr.isel(frame=idx).max('frame')


def local_max(fm, wnd):
    fm_max = maximum_filter(fm, wnd)
    return (fm == fm_max).astype(np.uint8)


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
        varr_sub.chunk(dict(spatial=50, frame=-1)),
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


def initialize(varr, seeds, thres_corr=0.8, wnd=10, chk=None):
    print("creating parallel schedule")
    harr, warr = seeds['height'].values, seeds['width'].values
    res_ls = [init_perseed(varr, h, w, wnd, thres_corr) for h, w in zip(harr, warr)]
    print("computing rois")
    res_ls = dask.compute(res_ls)[0]
    print("concatenating results")
    A = (xr.concat([r[0] for r in res_ls], 'unit_id')
         .assign_coords(unit_id = np.arange(len(res_ls)))
         .fillna(0))
    C = (xr.concat([r[1] for r in res_ls], 'unit_id')
         .assign_coords(unit_id = np.arange(len(res_ls))))
    print("initializing backgrounds")
    if not chk:
        chk = dict(height='auto', width='auto', frame='auto', unit_id='auto')
    A = A.reindex_like(varr.isel(frame=0)).fillna(0)
    A = A.chunk(dict(height=chk['height'], width=chk['width'], unit_id=-1))
    C = C.chunk(dict(frame=chk['frame'], unit_id=-1))
    varr = varr.chunk(dict(frame=chk['frame'], height=chk['height'], width=chk['width']))
    AC = xr.apply_ufunc(
        da.dot, A, C,
        input_core_dims=[['height', 'width', 'unit_id'], ['unit_id', 'frame']],
        output_core_dims=[['height', 'width', 'frame']],
        dask='allowed',
        output_dtypes=[A.dtype])
    Yr = varr - AC
    b = (Yr.chunk(dict(frame=-1, height=chk['height'], width=chk['width']))
         .mean('frame').compute())
    f = (Yr.chunk(dict(frame=chk['frame'], height=-1, width=-1))
         .mean('height').mean('width').compute())
    return A, C, b, f


def init_perseed(varr, h, w, wnd, thres_corr):
    h_sur, w_sur = (slice(h - wnd, h + wnd),
                    slice(w - wnd, w + wnd))
    sur = varr.sel(height=h_sur, width=w_sur)
    sur_flt = sur.stack(spatial=['height', 'width'])
    sp_idxs = sur_flt.coords['spatial'].values
    corr = xr.apply_ufunc(
        da.corrcoef,
        sur_flt,
        input_core_dims=[['spatial', 'frame']],
        output_core_dims=[['spatial', 'spatial_cp']],
        dask='allowed',
        output_sizes=dict(spatial_cp=len(sp_idxs)))
    sd_id = np.ravel_multi_index(
        (h - sp_idxs[0][0], w - sp_idxs[0][1]),
        (sur.sizes['height'], sur.sizes['width']))
    corr = (corr.isel(spatial_cp=sd_id)
            .squeeze().unstack('spatial'))
    mask = corr > thres_corr
    mask_lb = xr.apply_ufunc(da_label, mask, dask='allowed')
    sd_lb = mask_lb.sel(height=h, width=w)
    mask = (mask_lb == sd_lb)
    sur = sur.where(mask, 0)
    sd = sur.sel(height=h, width=w)
    A = xr.apply_ufunc(
        da.dot, sur, sd,
        input_core_dims=[['height', 'width', 'frame'], ['frame']],
        output_core_dims=[['height', 'width']],
        dask='allowed')
    A = A / da.linalg.norm(sd.data)
    A = A / da.linalg.norm(A.data)
    C = xr.apply_ufunc(
        da.tensordot, sur, A,
        input_core_dims=[['frame', 'height', 'width'], ['height', 'width']],
        output_core_dims=[['frame']],
        kwargs=dict(axes=[(1, 2), (0, 1)]),
        dask='allowed')
    return A, C


@da.as_gufunc(signature="(h, w)->(h, w)", output_dtypes=int, allow_rechunk=True)
def da_label(im):
    return label(im)[0]
