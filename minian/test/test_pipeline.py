import pytest

# Setting up

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import gc
import psutil
import numpy as np
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
import bokeh.plotting as bpl
import dask.array as da
import pandas as pd
import dask
import datashader as ds
import itertools as itt
import papermill as pm
import ast
import functools as fct
from holoviews.operation.datashader import datashade, regrid, dynspread
from datashader.colors import Sets1to3
from dask.diagnostics import ProgressBar
from IPython.core.display import display, HTML
import numpy as np;
import shutil

#Set up Initial Basic Parameters#
minian_path = "."
dpath = "./minian/test/test_movie"
dpath_fixture = "./minian/test/test_movie_fixture"
subset = dict(frame=slice(0,None))
subset_mc = None
interactive = True
output_size = 100
param_save_minian = {
    'dpath': dpath,
    'fname': 'minian',
    'backend': 'zarr',
    'meta_dict': dict(session_id=-1, session=-2, animal=-3),
    'overwrite': True}

#Pre-processing Parameters#
param_load_videos = {
    'pattern': 'msCam[0-9]+\.avi$',
    'dtype': np.uint8,
    'downsample': dict(frame=2,height=1,width=1),
    'downsample_strategy': 'subset'}
param_denoise = {
    'method': 'median',
    'ksize': 7}
param_background_removal = {
    'method': 'tophat',
    'wnd': 15}

#Motion Correction Parameters#
subset_mc = None
param_estimate_shift = {
    'dim': 'frame',
    'max_sh': 20}

#Initialization Parameters#
param_seeds_init = {
    'wnd_size': 1000,
    'method': 'rolling',
    'stp_size': 500,
    'nchunk': 100,
    'max_wnd': 15,
    'diff_thres': 2}
param_pnr_refine = {
    'noise_freq': 0.1,
    'thres': 1,
    'med_wnd': None}
param_ks_refine = {
    'sig': 0.05}
param_seeds_merge = {
    'thres_dist': 5,
    'thres_corr': 0.7,
    'noise_freq': 0.1}
param_initialize = {
    'thres_corr': 0.8,
    'wnd': 15,
    'noise_freq': 0.1}

#CNMF Parameters#
param_get_noise = {
    'noise_range': (0.1, 0.5),
    'noise_method': 'logmexp'}
param_first_spatial = {
    'dl_wnd': 15,
    'sparse_penal': 0.1,
    'update_background': True,
    'normalize': True,
    'zero_thres': 'eps'}
param_first_temporal = {
    'noise_freq': 0.1,
    'sparse_penal': 0.05,
    'p': 1,
    'add_lag': 20,
    'use_spatial': False,
    'jac_thres': 0.2,
    'zero_thres': 1e-8,
    'max_iters': 200,
    'use_smooth': True,
    'scs_fallback': False,
    'post_scal': True}
param_first_merge = {
    'thres_corr': 0.8}
param_second_spatial = {
    'dl_wnd': 15,
    'sparse_penal': 0.005,
    'update_background': True,
    'normalize': True,
    'zero_thres': 'eps'}
param_second_temporal = {
    'noise_freq': 0.1,
    'sparse_penal': 0.05,
    'p': 1,
    'add_lag': 20,
    'use_spatial': False,
    'jac_thres': 0.2,
    'zero_thres': 1e-8,
    'max_iters': 500,
    'use_smooth': True,
    'scs_fallback': False,
    'post_scal': True}

sys.path.append(minian_path)
from minian.utilities import load_videos, open_minian, save_minian, get_optimal_chk, rechunk_like
from minian.preprocessing import denoise, remove_background
from minian.motion_correction import estimate_shifts, apply_shifts
from minian.initialization import seeds_init, gmm_refine, pnr_refine, intensity_refine, ks_refine, seeds_merge, initialize
from minian.cnmf import get_noise_fft, update_spatial, compute_trace, update_temporal, unit_merge, smooth_sig
from minian.visualization import VArrayViewer, CNMFViewer, generate_videos, visualize_preprocess, visualize_seeds, visualize_gmm_fit, visualize_spatial_update, visualize_temporal_update, write_video

dpath = os.path.abspath(dpath)
hv.notebook_extension('bokeh')
if interactive:
    pbar = ProgressBar(minimum=2)
    pbar.register()

# Pre-processing
varr = load_videos(dpath, **param_load_videos)
chk = get_optimal_chk(varr.astype(float), dim_grp=[('frame',), ('height', 'width')])

def test_preprocessing():
    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(varr, framerate=5, summary=None)
        display(vaviewer.show())

    if interactive:
        try:
            subset_mc = list(vaviewer.mask.values())[0]
        except IndexError:
            pass

    varr_ref = varr.sel(subset)

    varr_min = varr_ref.min('frame').compute()
    varr_ref = varr_ref - varr_min
    
    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(
            [varr.rename('original'), varr_ref.rename('glow_removed')],
            framerate=5,
            summary=None,
            layout=True)
        display(vaviewer.show())

    hv.output(size=output_size)
    if interactive:
        display(visualize_preprocess(varr_ref.isel(frame=0), denoise, method=['median'], ksize=[5, 7, 9]))

    varr_ref = denoise(varr_ref, **param_denoise)
    
    hv.output(size=output_size)
    if interactive:
        display(visualize_preprocess(varr_ref.isel(frame=0), remove_background, method=['tophat'], wnd=[10, 15, 20]))

    varr_ref = remove_background(varr_ref, **param_background_removal)
    
    varr_ref = varr_ref.chunk(chk)
    
    varr_ref = save_minian(varr_ref.rename('org'), **param_save_minian)
    test_data_varr_ref = open_minian(dpath_fixture,
                      fname=param_save_minian['fname'],
                      backend=param_save_minian['backend'])['org']

    assert xr.testing.assert_equal(test_data_varr_ref, varr_ref), "Test Fail: arrays are not the same";
    
# motion correction
def test_motion_correction():
    varr_ref = open_minian(dpath,
                        fname=param_save_minian['fname'],
                        backend=param_save_minian['backend'])['org']

    shifts = estimate_shifts(varr_ref.sel(subset_mc), **param_estimate_shift)

    shifts = shifts.chunk(dict(frame=chk['frame'])).rename('shifts')
    shifts = save_minian(shifts, **param_save_minian)

    hv.output(size=output_size)
    if interactive:
        display(hv.NdOverlay(dict(width=hv.Curve(shifts.sel(variable='width')),
                                height=hv.Curve(shifts.sel(variable='height')))))

    Y = apply_shifts(varr_ref, shifts)
    Y = Y.fillna(0).astype(varr_ref.dtype)

    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(
            [varr_ref.rename('before_mc'), Y.rename('after_mc')],
            framerate=5,
            summary=None,
            layout=True)
        display(vaviewer.show())

    Y = Y.chunk(chk)
    Y = save_minian(Y.rename('Y'), **param_save_minian)

    vid_arr = xr.concat([varr_ref, Y], 'width').chunk(dict(height=-1, width=-1))
    vmax = varr_ref.max().compute().values
    write_video(vid_arr / vmax * 255, 'minian_mc.mp4', dpath)

    im_opts = dict(frame_width=500, aspect=752/480, cmap='Viridis', colorbar=True)
    (regrid(hv.Image(varr_ref.max('frame').compute(), ['width', 'height'], label='before_mc')).opts(**im_opts)
    + regrid(hv.Image(Y.max('frame').compute(), ['width', 'height'], label='after_mc')).opts(**im_opts))

    assert os.path.exists(os.path.join(dpath, "minian_mc.mp4")) == True, "minian_mc.mp4 was written to local folder"
    assert os.path.getsize(os.path.join(dpath, "minian_mc.mp4"))/(1024*1024) > 2, "minian_mc.mp4 was created and is at least 2 MB"
    
    # Remove "minian_mc.mp4" file after test is done
    os.remove(os.path.join(dpath, "minian_mc.mp4"))
    
# initialization
def test_initialization():
    minian = open_minian(dpath,
                        fname=param_save_minian['fname'],
                        backend=param_save_minian['backend'])
    
    Y = minian['Y'].astype(np.float)
    max_proj = Y.max('frame').compute()
    Y_flt = Y.stack(spatial=['height', 'width'])
    
    seeds = seeds_init(Y, **param_seeds_init)

    assert len(seeds) == 688, "Original 'seeds' array contains 688 elements"
    
    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds)

    if interactive:
        noise_freq_list = [0.005, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.45]
        example_seeds = seeds.sample(6, axis='rows')
        example_trace = (Y_flt
                        .sel(spatial=[tuple(hw) for hw in example_seeds[['height', 'width']].values])
                        .assign_coords(spatial=np.arange(6))
                        .rename(dict(spatial='seed')))
        smooth_dict = dict()
        for freq in noise_freq_list:
            trace_smth_low = smooth_sig(example_trace, freq)
            trace_smth_high = smooth_sig(example_trace, freq, btype='high')
            trace_smth_low = trace_smth_low.compute()
            trace_smth_high = trace_smth_high.compute()
            hv_trace = hv.HoloMap({
                'signal': (hv.Dataset(trace_smth_low)
                        .to(hv.Curve, kdims=['frame'])
                        .opts(frame_width=300, aspect=2, ylabel='Signal (A.U.)')),
                'noise': (hv.Dataset(trace_smth_high)
                        .to(hv.Curve, kdims=['frame'])
                        .opts(frame_width=300, aspect=2, ylabel='Signal (A.U.)'))
            }, kdims='trace').collate()
            smooth_dict[freq] = hv_trace

    hv.output(size=output_size)
    if interactive:
        hv_res = (hv.HoloMap(smooth_dict, kdims=['noise_freq']).collate().opts(aspect=2)
                .overlay('trace').layout('seed').cols(3))
        display(hv_res)

    seeds, pnr, gmm = pnr_refine(Y_flt, seeds.copy(), **param_pnr_refine)
    
    if gmm:
        display(visualize_gmm_fit(pnr, gmm, 100))
    
    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds, 'mask_pnr')
    
    assert len(seeds["mask_pnr"]) == 688, "Seeds array added new column 'mask_pnr' with 688 elements"
    
    seeds = ks_refine(Y_flt, seeds[seeds['mask_pnr']], **param_ks_refine)
    
    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds, 'mask_ks')

    seeds_final = seeds[seeds['mask_ks']].reset_index(drop=True)
    seeds_mrg = seeds_merge(Y_flt, seeds_final, **param_seeds_merge)

    assert len(seeds_final) == 521, 'Seeds array merged'    
    assert len(seeds_final['mask_mrg']) == 521, 'Seeds array contain mask_mrg column after merge'
    assert len(seeds_final['mask_ks']) == 521, 'Seeds array contain mask_ks column after merge'
        
    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds_mrg, 'mask_mrg')
    
    # assert seeds_final['mask_mrg'] != None

    A, C, b, f = initialize(Y, seeds_mrg[seeds_mrg['mask_mrg']], **param_initialize)
    
    assert A.name == 'Y', 'Variable A before renamed'
    assert C.name == 'Y', 'Variable C before renamed'
    assert b.name == 'Y', 'Variable b before renamed'
    assert f.name == 'Y', 'Variable f before renamed'

    im_opts = dict(frame_width=500, aspect=A.sizes['width']/A.sizes['height'], cmap='Viridis', colorbar=True)
    cr_opts = dict(frame_width=750, aspect=1.5*A.sizes['width']/A.sizes['height'])
    (regrid(hv.Image(A.sum('unit_id').rename('A').compute(), kdims=['width', 'height'])).opts(**im_opts)
    + regrid(hv.Image(C.rename('C').compute(), kdims=['frame', 'unit_id'])).opts(cmap='viridis', colorbar=True, **cr_opts)
    + regrid(hv.Image(b.rename('b').compute(), kdims=['width', 'height'])).opts(**im_opts)
    + datashade(hv.Curve(f.rename('f').compute(), kdims=['frame']), min_alpha=200).opts(**cr_opts)
    ).cols(2)
    
    A = save_minian(A.rename('A_init').rename(unit_id='unit_id_init'), **param_save_minian)
    C = save_minian(C.rename('C_init').rename(unit_id='unit_id_init'), **param_save_minian)
    b = save_minian(b.rename('b_init'), **param_save_minian)
    f = save_minian(f.rename('f_init'), **param_save_minian)
    
    assert A.name == 'A_init', 'Variable A renamed'
    assert C.name == 'C_init', 'Variable C renamed'
    assert b.name == 'b_init', 'Variable b renamed'
    assert f.name == 'f_init', 'Variable f renamed'
    
# # CNMF
def test_cnmf():
    minian = open_minian(dpath,
                        fname=param_save_minian['fname'],
                        backend=param_save_minian['backend'])
    Y = minian['Y'].astype(np.float)
    A_init = minian['A_init'].rename(unit_id_init='unit_id')
    C_init = minian['C_init'].rename(unit_id_init='unit_id')
    b_init = minian['b_init']
    f_init = minian['f_init']

    sn_spatial = get_noise_fft(Y, **param_get_noise).persist()

    if interactive:
        units = np.random.choice(A_init.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_init.sel(unit_id=units).persist()
        C_sub = C_init.sel(unit_id=units).persist()

    if interactive:
        sprs_ls = [0.05, 0.1, 0.5]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_b, cur_C, cur_f = update_spatial(
                Y, A_sub, b_init, C_sub, f_init,
                sn_spatial, dl_wnd=param_first_spatial['dl_wnd'], sparse_penal=cur_sprs)
            if cur_A.sizes['unit_id']:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = cur_C.compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=['sparse penalty'])

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    A_spatial, b_spatial, C_spatial, f_spatial = update_spatial(
        Y, A_init, b_init, C_init, f_init, sn_spatial, **param_first_spatial)

    hv.output(size=output_size)
    opts = dict(plot=dict(height=A_init.sizes['height'], width=A_init.sizes['width'], colorbar=True), style=dict(cmap='Viridis'))
    (regrid(hv.Image(A_init.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints Initial")
    + regrid(hv.Image((A_init.fillna(0) > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints Initial")
    + regrid(hv.Image(A_spatial.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints First Update")
    + regrid(hv.Image((A_spatial > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints First Update")).cols(2)

    hv.output(size=output_size)
    opts_im = dict(plot=dict(height=b_init.sizes['height'], width=b_init.sizes['width'], colorbar=True), style=dict(cmap='Viridis'))
    opts_cr = dict(plot=dict(height=b_init.sizes['height'], width=b_init.sizes['height'] * 2))
    (regrid(hv.Image(b_init.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial Initial')
    + datashade(hv.Curve(f_init.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal Initial')
    + regrid(hv.Image(b_spatial.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial First Update')
    + datashade(hv.Curve(f_spatial.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal First Update')
    ).cols(2)

    if interactive:
        units = np.random.choice(A_spatial.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_spatial.sel(unit_id=units).persist()
        C_sub = C_spatial.sel(unit_id=units).persist()

    if interactive:
        p_ls = [1]
        sprs_ls = [0.01, 0.05, 0.1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = compute_trace(Y, A_sub, b_spatial, C_sub, f_spatial).persist()
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(p_ls, sprs_ls, add_ls, noise_ls):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print("p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}"
                .format(cur_p, cur_sprs, cur_add, cur_noise))
            YrA, cur_C, cur_S, cur_B, cur_C0, cur_sig, cur_g, cur_scal = update_temporal(
                Y, A_sub, b_spatial, C_sub, f_spatial, sn_spatial, YrA=YrA,
                sparse_penal=cur_sprs, p=cur_p, use_spatial=False, use_smooth=True,
                add_lag = cur_add, noise_freq=cur_noise)
            YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (
                YrA.compute(), cur_C.compute(), cur_S.compute(), cur_g.compute(), cur_sig.compute(), A_sub.compute())
        hv_res = visualize_temporal_update(
            YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict,
            kdims=['p', 'sparse penalty', 'additional lag', 'noise frequency'])

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    YrA, C_temporal, S_temporal, B_temporal, C0_temporal, sig_temporal, g_temporal, scale = update_temporal(
        Y, A_spatial, b_spatial, C_spatial, f_spatial, sn_spatial, **param_first_temporal)
    A_temporal = A_spatial.sel(unit_id = C_temporal.coords['unit_id'])

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)
    (regrid(hv.Image(C_init.compute().rename('ci'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace Initial")
    + hv.Div('')
    + regrid(hv.Image(C_temporal.compute().rename('c1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace First Update")
    + regrid(hv.Image(S_temporal.compute().rename('s1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Spikes First Update")
    ).cols(2)


    # TODO Jesus: may be do some tests halfway the procedure?
    # TODO Jesus: do some testing with assertions


    hv.output(size=output_size)
    if interactive:
        h, w = A_spatial.sizes['height'], A_spatial.sizes['width']
        im_opts = dict(aspect=w/h, frame_width=500, cmap='Viridis')
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = list(set(A_spatial.coords['unit_id'].values) - set(A_temporal.coords['unit_id'].values))
        bad_units.sort()
        if len(bad_units)>0:
            hv_res = (hv.NdLayout({
                "Spatial Footprin": regrid(hv.Dataset(A_spatial.sel(unit_id=bad_units).compute().rename('A'))
                                        .to(hv.Image, kdims=['width', 'height'])).opts(**im_opts),
                "Spatial Footprints of Accepted Units": regrid(hv.Image(A_temporal.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**im_opts)
            })
                    + datashade(hv.Dataset(YrA.sel(unit_id=bad_units).rename('raw'))
                                .to(hv.Curve, kdims=['frame'])).opts(**cr_opts).relabel("Temporal Trace")).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

    hv.output(size=output_size)
    if interactive:
        display(visualize_temporal_update(YrA.compute(), C_temporal.compute(), S_temporal.compute(), 
                                        g_temporal.compute(), sig_temporal.compute(), A_temporal.compute()))

    A_mrg, sig_mrg, add_list = unit_merge(A_temporal, sig_temporal, [S_temporal, C_temporal], **param_first_merge)
    S_mrg, C_mrg = add_list[:]

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)
    (regrid(hv.Image(sig_temporal.compute().rename('c1'), kdims=['frame', 'unit_id'])).relabel("Temporal Signals Before Merge").opts(**opts_im) +
    regrid(hv.Image(sig_mrg.compute().rename('c2'), kdims=['frame', 'unit_id'])).relabel("Temporal Signals After Merge").opts(**opts_im))

    if interactive:
        units = np.random.choice(A_mrg.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_mrg.sel(unit_id=units).persist()
        sig_sub = sig_mrg.sel(unit_id=units).persist()

    if interactive:
        sprs_ls = [0.001, 0.005, 0.01]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_b, cur_C, cur_f = update_spatial(
                Y, A_sub, b_init, sig_sub, f_init,
                sn_spatial, dl_wnd=param_second_spatial['dl_wnd'], sparse_penal=cur_sprs)
            if cur_A.sizes['unit_id']:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = cur_C.compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=['sparse penalty'])

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    A_spatial_it2, b_spatial_it2, C_spatial_it2, f_spatial_it2 = update_spatial(
        Y, A_mrg, b_spatial, sig_mrg, f_spatial, sn_spatial, **param_second_spatial)

    hv.output(size=output_size)
    opts = dict(aspect=A_spatial_it2.sizes['width']/A_spatial_it2.sizes['height'], frame_width=500, colorbar=True, cmap='Viridis')
    (regrid(hv.Image(A_mrg.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints First Update")
    + regrid(hv.Image((A_mrg.fillna(0) > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints First Update")
    + regrid(hv.Image(A_spatial_it2.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints Second Update")
    + regrid(hv.Image((A_spatial_it2 > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints Second Update")).cols(2)

    hv.output(size=output_size)
    opts_im = dict(aspect=b_spatial_it2.sizes['width'] / b_spatial_it2.sizes['height'], frame_width=500, colorbar=True, cmap='Viridis')
    opts_cr = dict(aspect=2, frame_height=int(500 * b_spatial_it2.sizes['height'] / b_spatial_it2.sizes['width']))
    (regrid(hv.Image(b_spatial.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial First Update')
    + datashade(hv.Curve(f_spatial.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal First Update')
    + regrid(hv.Image(b_spatial_it2.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial Second Update')
    + datashade(hv.Curve(f_spatial_it2.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal Second Update')
    ).cols(2)

    if interactive:
        units = np.random.choice(A_spatial_it2.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_spatial_it2.sel(unit_id=units).persist()
        C_sub = C_spatial_it2.sel(unit_id=units).persist()

    if interactive:
        p_ls = [1]
        sprs_ls = [0.01, 0.05, 0.1]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = compute_trace(Y, A_sub, b_spatial, C_sub, f_spatial).persist()
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(p_ls, sprs_ls, add_ls, noise_ls):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print("p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}"
                .format(cur_p, cur_sprs, cur_add, cur_noise))
            YrA, cur_C, cur_S, cur_B, cur_C0, cur_sig, cur_g, cur_scal = update_temporal(
                Y, A_sub, b_spatial, C_sub, f_spatial, sn_spatial, YrA=YrA,
                sparse_penal=cur_sprs, p=cur_p, use_spatial=False, use_smooth=True,
                add_lag = cur_add, noise_freq=cur_noise)
            YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (
                YrA.compute(), cur_C.compute(), cur_S.compute(), cur_g.compute(), cur_sig.compute(), 
                A_sub.compute())
        hv_res = visualize_temporal_update(
            YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict,
            kdims=['p', 'sparse penalty', 'additional lag', 'noise frequency'])

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    YrA, C_temporal_it2, S_temporal_it2, B_temporal_it2, C0_temporal_it2, sig_temporal_it2, g_temporal_it2, scale_temporal_it2 = update_temporal(
        Y, A_spatial_it2, b_spatial_it2, C_spatial_it2, f_spatial_it2, sn_spatial, **param_second_temporal)
    A_temporal_it2 = A_spatial_it2.sel(unit_id=C_temporal_it2.coords['unit_id'])
    g_temporal_it2 = g_temporal_it2.sel(unit_id=C_temporal_it2.coords['unit_id'])
    A_temporal_it2 = rechunk_like(A_temporal_it2, A_spatial_it2)
    g_temporal_it2 = rechunk_like(g_temporal_it2, C_temporal_it2)

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)
    (regrid(hv.Image(C_mrg.compute().rename('c1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace First Update")
    + regrid(hv.Image(S_mrg.compute().rename('s1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Spikes First Update")
    + regrid(hv.Image(C_temporal_it2.compute().rename('c2').rename(unit_id='unit_id_it2'), kdims=['frame', 'unit_id_it2'])).opts(**opts_im).relabel("Temporal Trace Second Update")
    + regrid(hv.Image(S_temporal_it2.compute().rename('s2').rename(unit_id='unit_id_it2'), kdims=['frame', 'unit_id_it2'])).opts(**opts_im).relabel("Spikes Second Update")).cols(2)

    hv.output(size=output_size)
    if interactive:
        h, w = A_spatial_it2.sizes['height'], A_spatial_it2.sizes['width']
        im_opts = dict(aspect=w/h, frame_width=500, cmap='Viridis')
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = list(set(A_spatial_it2.coords['unit_id'].values) - set(A_temporal_it2.coords['unit_id'].values))
        bad_units.sort()
        if len(bad_units)>0:
            hv_res = (hv.NdLayout({
                "Spatial Footprin": regrid(hv.Dataset(A_spatial_it2.sel(unit_id=bad_units).compute().rename('A'))
                                        .to(hv.Image, kdims=['width', 'height'])).opts(**im_opts),
                "Spatial Footprints of Accepted Units": regrid(hv.Image(A_temporal_it2.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**im_opts)
            })
                    + datashade(hv.Dataset(YrA.sel(unit_id=bad_units).compute().rename('raw'))
                                .to(hv.Curve, kdims=['frame'])).opts(**cr_opts).relabel("Temporal Trace")).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

    hv.output(size=output_size)
    if interactive:
        display(visualize_temporal_update(YrA.compute(), C_temporal_it2.compute(), S_temporal_it2.compute(), 
                                        g_temporal_it2.compute(), sig_temporal_it2.compute(), 
                                        A_temporal_it2.compute()))

    A_temporal_it2 = save_minian(A_temporal_it2.rename('A'), **param_save_minian)
    C_temporal_it2 = save_minian(C_temporal_it2.rename('C'), **param_save_minian)
    S_temporal_it2 = save_minian(S_temporal_it2.rename('S'), **param_save_minian)
    g_temporal_it2 = save_minian(g_temporal_it2.rename('g'), **param_save_minian)
    C0_temporal_it2 = save_minian(C0_temporal_it2.rename('C0'), **param_save_minian)
    B_temporal_it2 = save_minian(B_temporal_it2.rename('bl'), **param_save_minian)
    b_spatial_it2 = save_minian(b_spatial_it2.rename('b'), **param_save_minian)
    f_spatial_it2 = save_minian(f_spatial_it2.rename('f'), **param_save_minian)

    minian = open_minian(dpath,
                        fname=param_save_minian['fname'],
                        backend=param_save_minian['backend'])
    varr = load_videos(dpath, **param_load_videos)
    chk = get_optimal_chk(varr.astype(float), dim_grp=[('frame',), ('height', 'width')])
    varr = varr.chunk(dict(frame=chk['frame']))

    generate_videos(
        minian, varr, dpath, param_save_minian['fname'] + ".mp4", scale='auto')

    if interactive:
        cnmfviewer = CNMFViewer(minian)

    hv.output(size=output_size)
    if interactive:
        display(cnmfviewer.show())

    if interactive:
        save_minian(cnmfviewer.unit_labels, **param_save_minian)

    assert os.path.exists(os.path.join(dpath, "minian.mp4")) == True, "minian.mp4 was written to local folder"
    assert os.path.getsize(os.path.join(dpath, "minian.mp4"))/(1024*1024) >= 2, "minian.mp4 is at least 2 MB"
    
    # Remove create 'minian.mp4' file and 'minian' folder, this is needed so when test is run again it doesn't test
    # using the files from the previous test run
    os.remove(os.path.join(dpath, "minian.mp4"))
    shutil.rmtree(os.path.join(dpath, "minian"))
