import numpy as np
import xarray as xr
import pandas as pd
import dask
import pyfftw.interfaces.numpy_fft as npfft
import graph_tool.all as gt
import dask.array.fft as dafft
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
from scipy.stats import zscore, kstest
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
from IPython.core.debugger import set_trace
from scipy.signal import butter, lfilter


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
        kwargs=dict(wnd=max_wnd)).sum('sample')
    with ProgressBar():
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


def gmm_refine(varr, seeds, q=(0.1, 99.9)):
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
    with ProgressBar():
        varr_pv = varr_pv.compute()
    print("fitting GMM models")
    dat = varr_pv.values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(dat)
    idg = gmm.means_.argmax()
    idx_valid = gmm.predict(dat) == idg
    seeds['mask_gmm'] = idx_valid
    return seeds


def pnr_refine(varr, seeds, noise_freq=0.25, thres=1.5):
    print("selecting seeds") 
    varr_sub = varr.sel(
        spatial=[tuple(hw) for hw in seeds[['height', 'width']].values])
    varr_sub = varr_sub.chunk(dict(frame=-1, spatial='auto'))
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
    varr_sub_ptp = xr.apply_ufunc(
        np.ptp,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[['frame']],
        dask='parallelized',
        vectorize=True,
        output_dtypes=[varr_sub.dtype]).compute()
    varr_noise_ptp = xr.apply_ufunc(
        np.ptp,
        varr_noise.chunk(dict(frame=-1)).real,
        input_core_dims=[['frame']],
        dask='parallelized',
        vectorize=True,
        output_dtypes=[varr_sub.dtype]).compute()
    pnr = varr_sub_ptp / varr_noise_ptp
    mask = pnr > thres
    with ProgressBar():
        mask = mask.compute()
    mask_df = mask.to_pandas().rename('mask_pnr').reset_index()
    seeds = pd.merge(seeds, mask_df, on=['height', 'width'], how='left')
    return seeds, pnr


def intensity_refine(varr, seeds):
    try:
        fm_max = varr.max('frame')
    except ValueError:
        print("using input as max projection")
        fm_max = varr
    bins = np.around(
        fm_max.sizes['height'] * fm_max.sizes['width'] / 10).astype(int)
    hist, edges = np.histogram(fm_max, bins=bins)
    try:
        thres = edges[np.argmax(hist) * 2]
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


def seeds_merge(varr, seeds, thres_dist=5, thres_corr=0.6):
    varr_sub = (varr.where(seeds > 0)
                .stack(sample=('height', 'width'))
                .dropna('sample', how='all'))
    seeds_ref = (seeds.where(seeds > 0)
                 .stack(sample=('height', 'width'))
                 .dropna('sample', how='all'))
    varr_max = varr.max('frame').where(seeds > 0).stack(
        sample=('height', 'width')).dropna('sample').compute()
    crds = seeds_ref.coords
    nsmp = len(crds['sample'])
    hwarr = xr.concat([crds['height'], crds['width']], dim='dim')
    dist = xr.apply_ufunc(
        lambda a: squareform(pdist(a)),
        hwarr,
        input_core_dims=[['sample', 'dim']],
        output_core_dims=[['sampleA', 'sampleB']],
        dask='parallelized',
        output_dtypes=[float],
        output_sizes=dict(sampleA=nsmp, sampleB=nsmp)).assign_coords(
            sampleA=np.arange(len(crds['sample'])),
            sampleB=np.arange(len(crds['sample'])))
    corr = xr.apply_ufunc(
        np.corrcoef,
        varr_sub.chunk(dict(sample=-1, frame=-1)),
        input_core_dims=[['sample', 'frame']],
        output_core_dims=[['sampleA', 'sampleB']],
        dask='parallelized',
        output_sizes=dict(sampleA=nsmp, sampleB=nsmp),
        output_dtypes=[float]).assign_coords(
            sampleA=np.arange(len(crds['sample'])),
            sampleB=np.arange(len(crds['sample'])))
    adj = np.logical_and(dist < thres_dist, corr > thres_corr)
    adj = adj.compute()
    np.fill_diagonal(adj.values, 0)
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


def initialize(varr, seeds, thres_corr=0.8, wnd=10, schd='processes', chk=None):
    print("reshaping video array")
    old_err = np.seterr(divide='raise')
    varr_flt = varr.stack(sample=('height', 'width'))
    if schd == 'threads':
        varr_flt = delayed(varr_flt)
    seeds_ref = (seeds.where(seeds > 0)
                 .stack(sample=('height', 'width'))
                 .dropna('sample', how='all'))
    res = []
    print("creating parallel schedule")
    for cur_crd, cur_sd in seeds_ref.groupby('sample'):
        cur_sur = (slice(cur_crd[0] - wnd, cur_crd[0] + wnd),
                   slice(cur_crd[1] - wnd, cur_crd[1] + wnd))
        sd = varr_flt.sel(sample=cur_crd)
        sur = varr_flt.sel(sample=cur_sur)
        cur_res = delayed(initialize_perseed)(cur_crd, sd, sur, thres_corr)
        # cur_res = initialize_perseed(cur_crd, sd, sur, thres_corr)
        res.append(cur_res)
    print("computing roi")
    with ProgressBar(), dask.config.set(scheduler=schd):
        res = compute(res)[0]
    print("concatenating results")
    A = xr.concat([r[0] for r in res],
                  'unit_id').assign_coords(unit_id=np.arange(len(res)))
    C = xr.concat([r[1] for r in res],
                  'unit_id').assign_coords(unit_id=np.arange(len(res)))
    A = A.reindex_like(varr).fillna(0)
    print("initializing backgrounds")
    if not chk:
        chk = dict(height='auto', width='auto', frame='auto', unit_id='auto')
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
    with ProgressBar():
        # Yr = Yr.compute()
        b = (Yr.chunk(dict(frame=-1, height=chk['height'], width=chk['width']))
             .mean('frame').compute())
        f = (Yr.chunk(dict(frame=chk['frame'], height=-1, width=-1))
             .mean('height').mean('width').compute())
    np.seterr(**old_err)
    return A, C, b, f


def initialize_perseed(sd_id, sd, sur, thres_corr):
    # sd = varr.sel(sample=sd_id)
    # sur = varr.sel(sample=sur_id)
    sur = sur.where(sur.std('frame') > 0, drop=True)
    smp_idxs = sur.coords['sample']
    sd_id_flt = np.nonzero([s == sd_id for s in smp_idxs.data])[0]
    corr = xr.apply_ufunc(
        np.corrcoef,
        sur,
        input_core_dims=[['sample', 'frame']],
        output_core_dims=[['sample', 'sample_cp']],
        output_sizes=dict(sample_cp=len(smp_idxs))).compute()
    corr = corr.isel(sample_cp=sd_id_flt).squeeze()
    mask = (corr > thres_corr).unstack('sample')
    mask = mask.where(mask > 0, drop=True).fillna(0)
    mask_lb = xr.apply_ufunc(
        lambda m: label(m)[0], mask).compute()
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


def initialize_new(varr, seeds, thres_cor, wnd=20):
    print("reshaping")
    varr_flt = varr.stack(sample=['height', 'width'])
    xr.apply_ufunc(da.corrcoef, varr_flt, input_core_dims=['sample', 'frame'], output_core_dims=['sample', 'sample_cp'])