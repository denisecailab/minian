import numpy as np
import xarray as xr
import pandas as pd
import dask
import pyfftw.interfaces.numpy_fft as npfft
import graph_tool.all as gt
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
from scipy.stats import zscore, kstest
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
from IPython.core.debugger import set_trace


def seeds_init(varr, wnd_size=500, method='rolling', stp_size=200, nchunk=100):
    idx_fm = varr.coords['frame']
    nfm = len(idx_fm.values)
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
    res = [dask.delayed(max_proj_frame)(varr, cur_idx) for cur_idx in max_idx]
    print("computing max projection")
    with ProgressBar():
        res = dask.compute(res)[0]
    print("concatenating samples")
    max_res = xr.concat(res, 'sample').chunk(dict(sample=10))
    print("calculating local maximum")
    loc_max = xr.apply_ufunc(
        local_max,
        max_res,
        input_core_dims=[['height', 'width']],
        output_core_dims=[['height', 'width']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.uint8],
        kwargs=dict(wnd=20)).sum('sample')
    with ProgressBar():
        loc_max = loc_max.compute()
    return loc_max


def max_proj_frame(varr, idx):
    return varr.isel(frame=idx).max('frame')


def local_max(fm, wnd):
    fm_max = maximum_filter(fm, wnd)
    return (fm == fm_max).astype(np.uint8)


def gmm_refine(varr, seeds):
    varr_sub = varr.where(seeds > 0).stack(sample=('height',
                                                   'width')).dropna('sample')
    seeds_ref = seeds.where(seeds > 0).stack(sample=('height',
                                                     'width')).dropna('sample')
    varr_pv = varr_sub.quantile(
        0.999, dim='frame') - varr_sub.quantile(
            0.001, dim='frame')
    dat = varr_pv.data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(dat)
    idg = gmm.means_.argmax()
    idx_valid = gmm.predict(dat) == idg
    seeds_ref = seeds_ref.isel(sample=idx_valid)
    return seeds_ref.unstack('sample').fillna(0)


def pnr_refine(varr, seeds, thres=1.5):
    varr_sub = varr.where(seeds > 0).stack(sample=('height',
                                                   'width')).dropna('sample')
    seeds_ref = seeds.where(seeds > 0).stack(sample=('height',
                                                     'width')).dropna('sample')
    varr_fft = xr.apply_ufunc(
        npfft.fft,
        varr_sub,
        input_core_dims=[['frame']],
        output_core_dims=[['frame']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.complex128])
    fidx = varr_fft.coords['frame']
    cut25, cut75 = np.around(fidx.quantile(0.25)), np.around(
        fidx.quantile(0.75))
    varr_fft.loc[dict(frame=slice(0, cut25))] = 0
    varr_fft.loc[dict(frame=slice(cut75, len(fidx)))] = 0
    varr_ifft = xr.apply_ufunc(
        npfft.ifft,
        varr_fft,
        input_core_dims=[['frame']],
        output_core_dims=[['frame']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.complex128])
    varr_sub_ptp = xr.apply_ufunc(
        np.ptp,
        varr_sub,
        input_core_dims=[['frame']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[varr_sub.dtype])
    varr_ifft_ptp = xr.apply_ufunc(
        np.ptp,
        varr_ifft.real,
        input_core_dims=[['frame']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[varr_sub.dtype])
    mask = (varr_sub_ptp / varr_ifft_ptp) > thres
    seeds_ref = seeds_ref.where(mask).dropna('sample')
    return seeds_ref.unstack('sample').fillna(0)


def intensity_refine(varr, seeds):
    fm_max = varr.max('frame')
    bins = np.around(
        fm_max.sizes['height'] * fm_max.sizes['width'] / 10).astype(int)
    hist, edges = np.histogram(fm_max, bins=bins)
    thres = edges[np.argmax(hist) * 2]
    seeds_ref = seeds.where(fm_max > thres).fillna(0)
    return seeds_ref


def ks_refine(varr, seeds, sig=0.05):
    varr_sub = varr.where(seeds > 0).stack(sample=('height',
                                                   'width')).dropna('sample')
    seeds_ref = seeds.where(seeds > 0).stack(sample=('height',
                                                     'width')).dropna('sample')
    ks = xr.apply_ufunc(
        lambda x: kstest(zscore(x), 'norm')[1],
        varr_sub,
        input_core_dims=[['frame']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float])
    seeds_ref = seeds_ref.where(ks < sig).dropna('sample')
    return seeds_ref.unstack('sample').fillna(0)


def seeds_merge(varr, seeds, thres_dist=5, thres_corr=0.6):
    varr_sub = varr.where(seeds > 0).stack(sample=('height',
                                                   'width')).dropna('sample')
    seeds_ref = seeds.where(seeds > 0).stack(sample=('height',
                                                     'width')).dropna('sample')
    varr_max = varr.max('frame').where(seeds > 0).stack(
        sample=('height', 'width')).dropna('sample')
    crds = seeds_ref.coords
    hwarr = xr.concat([crds['height'], crds['width']], dim='dim')
    dist = xr.apply_ufunc(
        lambda a: squareform(pdist(a)),
        hwarr,
        input_core_dims=[['sample', 'dim']],
        output_core_dims=[['sampleA', 'sampleB']],
        dask='parallelized',
        output_dtypes=[float]).assign_coords(
            sampleA=np.arange(len(crds['sample'])),
            sampleB=np.arange(len(crds['sample'])))
    corr = xr.apply_ufunc(
        np.corrcoef,
        varr_sub,
        input_core_dims=[['sample', 'frame']],
        output_core_dims=[['sampleA', 'sampleB']],
        dask='parallelized',
        output_dtypes=[float]).assign_coords(
            sampleA=np.arange(len(crds['sample'])),
            sampleB=np.arange(len(crds['sample'])))
    adj = np.logical_and(dist < thres_dist, corr > thres_corr)
    np.fill_diagonal(adj.data, 0)
    iso = adj.sum('sampleB')
    iso = iso.where(iso == 0).dropna('sampleA')
    adj = xr.apply_ufunc(np.triu, adj)
    eg_list = adj.to_dataframe(name='adj')
    eg_list = eg_list[eg_list['adj']].reset_index()[['sampleA', 'sampleB']]
    g = gt.Graph(directed=False)
    gmap = g.add_edge_list(eg_list.values, hashed=True)
    comp, hist = gt.label_components(g)
    seeds_final = set(iso.coords['sampleA'].data.tolist())
    for cur_cmp in np.unique(comp.a):
        cur_smp = [gmap[v] for v in np.where(comp.a == cur_cmp)[0]]
        cur_max = varr_max.isel(sample=cur_smp)
        max_seed = cur_smp[np.argmax(cur_max.data)]
        seeds_final.add(max_seed)
    seeds_ref = seeds_ref.isel(sample=list(seeds_final))
    return seeds_ref.unstack('sample').fillna(0)


def initialize(varr, seeds, thres_corr=0.9, wnd=20):
    old_err = np.seterr(divide='raise')
    varr_flt = varr.stack(sample=('height', 'width'))
    seeds_ref = seeds.where(seeds > 0).stack(sample=('height',
                                                     'width')).dropna('sample')
    res = []
    print("creating parallel schedule")
    for cur_crd, cur_sd in seeds_ref.groupby('sample'):
        cur_sur = (slice(cur_crd[0] - wnd, cur_crd[0] + wnd),
                   slice(cur_crd[1] - wnd, cur_crd[1] + wnd))
        cur_res = delayed(initialize_perseed)(varr_flt, cur_crd, cur_sur, thres_corr)
        res.append(cur_res)
    print("computing roi")
    with ProgressBar():
        res = compute(res)[0]
    print("concatenating results")
    A = xr.concat([r[0] for r in res],
                  'unit_id').assign_coords(unit_id=np.arange(len(res)))
    C = xr.concat([r[1] for r in res],
                  'unit_id').assign_coords(unit_id=np.arange(len(res)))
    np.seterr(**old_err)
    return A.reindex_like(varr), C


def initialize_perseed(varr, sd_id, sur_id, thres_corr):
    sd = varr.sel(sample=sd_id)
    sur = varr.sel(sample=sur_id)
    sd_id_flt = np.nonzero([s == sd_id for s in sur.coords['sample'].data])[0]
    sur = sur.where(sur.std('frame') > 0, drop=True)
    corr = xr.apply_ufunc(
        np.corrcoef,
        sur,
        input_core_dims=[['sample', 'frame']],
        output_core_dims=[['sample', 'sampleflt']],
        dask='forbidden').isel(sampleflt=sd_id_flt).squeeze()
    mask = (corr > thres_corr).unstack('sample')
    mask = mask.where(mask > 0, drop=True).fillna(0)
    mask_lb = xr.apply_ufunc(lambda m: label(m)[0], mask, dask='forbidden')
    sd_lb = mask_lb.sel(height=sd_id[0], width=sd_id[1])
    mask = mask_lb.where(
        mask_lb == sd_lb,
        drop=True).fillna(0).stack(sample=('height', 'width'))
    sur = sur.where(mask, drop=True)
    try:
        _A = sur.dot(sd, 'frame') / np.linalg.norm(sd)
        _A = _A / np.linalg.norm(_A)
    except FloatingPointError:
        set_trace()
    _C = sur.dot(_A, 'sample')
    return _A.unstack('sample').fillna(0), _C
