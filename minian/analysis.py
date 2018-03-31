import cv2
import caiman as cm
import numpy as np
import miniscope as ms
import os
import glob
import time
import pylab as pl
import scipy
import re
import scipy.ndimage as ndi
import h5py as h5
from caiman.source_extraction.cnmf import cnmf
from caiman import motion_correction
from caiman.utils import visualization
from caiman import components_evaluation
from skvideo import io as sio

params_movie = {
    # 'fname': 'Animal G95/Day7/H19_M30_S21/msCam2.tif',
    'fname': '/home/phild/Documents/sync/project/transfer/data/Animal G95/Day7/H19_M30_S21/msCam2.tif',
    'niter_rig': 1, # maximum number of iterations rigid motion correction,
    # in general is 1. 0 will quickly initialize a template with the first frames
    'max_shifts': (20, 20),  # maximum allow rigid shift
    'splits_rig': 28,  # for parallelization split the movies in  num_splits chuncks across time
    #  if none all the splits are processed and the movie is saved
    'num_splits_to_process_rig': None,
    # intervals at which patches are laid out for motion correction
    'strides': (48, 48),
    # overlap between pathes (size of patch strides+overlaps)
    'overlaps': (24, 24),
    'splits_els': 28,  # for parallelization split the movies in  num_splits chuncks across time
    # if none all the splits are processed and the movie is saved
    'num_splits_to_process_els': [14, None],
    'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
    # maximum deviation allowed for patch with respect to rigid
    # shift
    'max_deviation_rigid': 3,
    'p': 1,  # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allowed
    'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50
    'stride_cnmf': 6,  # amounpl.it of overlap between the patches in pixels
    'K': 4,  # number of components per patch
    # if dendritic. In this case you need to set init_method to
    # sparse_nmf
    'is_dendrites': False,
    'init_method': 'greedy_roi',
    'gSig': [4, 4],  # expected half size of neurons
    'alpha_snmf': None,  # this controls sparsity
    'final_frate': 30
}

# mov_orig = cm.load(params_movie['fname']).astype(np.float32)
dpath = '/media/share/Denise/Wired Valence/Wired Valence Organized Data/MS101/4/H11_M52_S45/'
dpattern = 'msCam*.avi'
dlist = sorted(glob.glob(dpath + dpattern),
               key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
vdlist = list()
for vname in dlist:
    vdlist.append(sio.vread(vname, as_grey=True))
mov_orig = cm.movie(np.squeeze(np.concatenate(vdlist, axis=0))).astype(np.float32)
# column correction
meanrow = np.mean(np.mean(mov_orig, 0), 0)
addframe = np.tile(meanrow, (mov_orig.shape[1], 1))
mov_cc = mov_orig - np.tile(addframe, (mov_orig.shape[0], 1, 1))
mov_cc = mov_cc - np.min(mov_cc)
# filter
mov_ft = mov_cc.copy()
for fid, fm in enumerate(mov_cc):
    mov_ft[fid] = ndi.uniform_filter(fm, 2) - ndi.uniform_filter(fm, 40)
mov_orig = (mov_orig - np.min(mov_orig)) / (np.max(mov_orig) - np.min(mov_orig))
mov_ft = (mov_ft - np.min(mov_ft)) / (np.max(mov_ft) - np.min(mov_ft))
np.save(dpath+'mov_orig', mov_orig)
np.save(dpath+'mov_ft', mov_ft)
del mov_orig, dlist, vdlist, mov_cc, mov_ft
# select roi
# mov_roi = mov_ft[:, 80:200, 474:700]
# start parallel
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
mc_data = motion_correction.MotionCorrect(
    dpath + 'mov_ft.npy', 0,
    dview=dview, max_shifts=params_movie['max_shifts'],
    niter_rig=params_movie['niter_rig'], splits_rig=params_movie['splits_rig'],
    num_splits_to_process_rig=params_movie['num_splits_to_process_rig'],
    strides=params_movie['strides'], overlaps=params_movie['overlaps'],
    splits_els=params_movie['splits_els'],
    num_splits_to_process_els=params_movie['num_splits_to_process_els'],
    upsample_factor_grid=params_movie['upsample_factor_grid'],
    max_deviation_rigid=params_movie['max_deviation_rigid'],
    shifts_opencv=True, nonneg_movie=False, roi=(80, 200, 474, 700))
mc_data.motion_correct_rigid(save_movie=True)
mov_rig = cm.load(mc_data.fname_tot_rig)
np.save(dpath+'mov_rig', mov_rig)
np.savez(dpath+'mc',
         templates_rig=mc_data.templates_rig,
         shifts_rig=mc_data.shifts_rig,
         total_templates_rig=mc_data.total_template_rig,
         max_shifts=mc_data.max_shifts,
         roi=mc_data.roi)
del mov_rig
# mc_data.motion_correct_pwrigid(template=mc_data.total_template_rig)
mov, dims, T = cm.load_memmap(mc_data.fname_tot_rig)
mov = np.reshape(mov.T, [T] + list(dims), order='F')
cnm = cnmf.CNMF(
    n_processes, k=params_movie['K'], gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'],
    p=params_movie['p'], dview=dview, Ain=None, rf=params_movie['rf'], stride=params_movie['stride_cnmf'],
    memory_fact=1, method_init=params_movie['init_method'], alpha_snmf=params_movie['alpha_snmf'], only_init_patch=True,
    gnb=1, method_deconvolution='oasis')
cnm = cnm.fit(mov)
# Cn = cm.local_correlations(mov_orig, swap_dim=False)
idx_comp, idx_comp_bad = components_evaluation.estimate_components_quality(
    cnm.C + cnm.YrA, np.reshape(mov, dims + (T,), order='F'), cnm.A, cnm.C, cnm.b,
    cnm.f, params_movie['final_frate'], Npeaks=10, r_values_min=.7, fitness_min=-40, fitness_delta_min=-40
)
# visualization.plot_contours(cnm.A.tocsc()[:, idx_comp], Cn)
A2 = cnm.A.tocsc()[:, idx_comp]
C2 = cnm.C[idx_comp]
cnm = cnmf.CNMF(
    n_processes, k=A2.shape, gSig=params_movie['gSig'], merge_thresh=params_movie['merge_thresh'], p=params_movie['p'],
    dview=dview, Ain=A2, Cin=C2, f_in=cnm.f, rf=None, stride=None, method_deconvolution='oasis')
cnm = cnm.fit(mov)
idx_comp, idx_comp_bad = components_evaluation.estimate_components_quality(
    cnm.C + cnm.YrA, np.reshape(mov, dims + (T,), order='F'), cnm.A, cnm.C, cnm.b,
    cnm.f, params_movie['final_frate'], Npeaks=10, r_values_min=.75, fitness_min=-50, fitness_delta_min=-50
)
# visualization.plot_contours(cnm.A.tocsc()[:, idx_comp], Cn)
cm.cluster.stop_server()
cnm.A = (cnm.A - np.min(cnm.A))/(np.max(cnm.A) - np.min(cnm.A))
cnm.C = (cnm.C - np.min(cnm.C))/(np.max(cnm.C) - np.min(cnm.C))
cnm.b = (cnm.b - np.min(cnm.b))/(np.max(cnm.b) - np.min(cnm.b))
cnm.f = (cnm.f - np.min(cnm.f))/(np.max(cnm.f) - np.min(cnm.f))
np.savez(dpath + 'cnm', A=cnm.A.todense(), C=cnm.C, b=cnm.b, f=cnm.f, YrA=cnm.YrA, sn=cnm.sn, dims=dims)
AC = (cnm.A.dot(cnm.C)).reshape(dims+(-1,), order='F').transpose([2, 0, 1])
ACmin = np.min(AC)
ACmax = np.max(AC)
AC = (AC - ACmin) / (ACmax - ACmin)
np.save(dpath + 'AC', AC)
del AC, ACmax, ACmin
ACbf = (cnm.A.dot(cnm.C) + cnm.b.dot(cnm.f)).reshape(dims+(-1,), order='F').transpose([2, 0, 1])
ACbfmin = np.min(ACbf)
ACbfmax = np.max(ACbf)
ACbf = (ACbf - ACbfmin) / (ACbfmax - ACbfmin)
np.save(dpath + 'ACbf', ACbf)
del ACbf, ACbfmax, ACbfmin
ms.save_video('/home/phild/MS101_6_result.mp4', dpath+'mov_orig.npy', dpath+'mov_rig.npy', dpath+'AC.npy', dpath+'ACbf.npy', dsratio=3)
os.remove(dpath+'mov_orig.npy')
os.remove(dpath+'mov_ft.npy')
# os.remove(mc_data.fname_tot_rig)
# os.remove(dpath+'mov_rig.npy')
os.remove(dpath+'AC.npy')
os.remove(dpath+'ACbf.npy')
