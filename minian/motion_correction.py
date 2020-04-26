import numpy as np
import xarray as xr
import cv2
import sys
import itertools as itt
import pyfftw.interfaces.numpy_fft as npfft
import numba as nb
import dask.array as darr
from scipy.stats import zscore
from scipy.ndimage import center_of_mass, label
from collections import OrderedDict
from skimage import transform as tf
from skimage.morphology import disk
from scipy.stats import zscore
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from IPython.core.debugger import set_trace
from .utilities import get_optimal_chk


def detect_and_correct_old(mov):
    surf = cv2.xfeatures2d.SURF_create(200)
    matcher = cv2.BFMatcher(crossCheck=True)
    detect_list = [surf.detectAndCompute(f, None) for f in mov]
    kp_list = [d[0] for d in detect_list]
    des_list = [d[1] for d in detect_list]
    match_list = []
    for des0, des1 in zip(des_list[:-1], des_list[1:]):
        match_list.append(matcher.match(des0, des1))
    matching_points = []
    for iframe, matches in enumerate(match_list):
        points0 = []
        points1 = []
        matches.sort(key=lambda ma: ma.distance, reverse=True)
        for ma in matches[:3]:
            points0.append(kp_list[iframe][ma.queryIdx].pt)
            points1.append(kp_list[iframe + 1][ma.trainIdx].pt)
        points0 = np.float32(np.array(points0))
        points1 = np.float32(np.array(points1))
        matching_points.append((points0, points1))
    trans_list = [
        cv2.getAffineTransform(pt[0], pt[1]) for pt in matching_points
    ]
    mov_correct = mov.copy()
    for iframe, trans in enumerate(trans_list):
        mov_correct[iframe + 1] = cv2.warpAffine(mov_correct[iframe], trans,
                                                 mov[0].shape[::-1])
    return mov_correct


def detect_and_correct(varray,
                       d_th=None,
                       r_th=None,
                       z_th=None,
                       q_th=None,
                       h_th=400,
                       std_thres=5,
                       opt_restr=5,
                       opt_std_thres=15,
                       opt_h_prop=0.1,
                       opt_err_thres=40,
                       method='translation',
                       upsample=None,
                       weight=False,
                       invert=False,
                       enhance=True):
    surf = cv2.xfeatures2d.SURF_create(h_th, extended=True)
    matcher = cv2.BFMatcher_create(crossCheck=True)
    # clache = cv2.createCLAHE(clipLimit=2, tileGridSize=(50, 50))
    varray = varray.transpose('frame', 'height', 'width')
    varr_mc = varray.astype(np.uint8)
    varr_ref = varray.astype(np.uint8)
    lk_params = dict(
        winSize=(200, 300),
        maxLevel=0,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,
                  0.0001),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    frm_idx = varr_mc.coords['frame']
    if invert:
        varr_ref = 255 - varr_ref
    if upsample:
        w = varray.coords['width']
        h = varray.coords['height']
        w_up = np.linspace(w[0], w[-1], len(w) * upsample)
        h_up = np.linspace(h[0], h[-1], len(h) * upsample)
        varr_ref = varr_ref.reindex(method='nearest', width=w_up, height=h_up)
    if enhance:
        for fid in frm_idx.values:
            fm = varr_ref.sel(frame=fid)
            fm.values = cv2.bilateralFilter(
                cv2.equalizeHist(fm.values), 9, 250, 250)
            varr_ref.loc[dict(frame=fid)] = fm
    match_dict = OrderedDict()
    shifts, shifts_idx, drop_idx = ([], [], [])
    last_registered = frm_idx[0].values
    for i, fid in enumerate(frm_idx[1:].values):
        im_src = varr_ref.sel(frame=last_registered).values
        im_dst = varr_ref.sel(frame=fid).values
        detect_src = surf.detectAndCompute(im_src, None)
        detect_dst = surf.detectAndCompute(im_dst, None)
        if not detect_src[0]:
            sys.stdout.write("\033[K")
            print("insufficient features for frame {}".format(last_registered))
            drop_idx.append(fid)
            continue
        if not detect_dst[0]:
            sys.stdout.write("\033[K")
            print("insufficient features for frame {}".format(fid))
            drop_idx.append(fid)
            continue
        match = matcher.match(detect_src[1], detect_dst[1])
        p_src, p_dst, eu_d, eu_x, eu_y, vma = ([], [], [], [], [], [])
        for idm, ma in enumerate(match):
            if True:
                pt0 = np.array(detect_src[0][ma.queryIdx].pt)
                pt1 = np.array(detect_dst[0][ma.trainIdx].pt)
                pt_diff = pt0 - pt1
                d = np.sqrt(np.sum(pt_diff**2))
                r = ma.distance
                if (d < d_th if d_th else True and r < r_th if r_th else True):
                    p_src.append(detect_src[0][ma.queryIdx].pt)
                    p_dst.append(detect_dst[0][ma.trainIdx].pt)
                    eu_d.append(d)
                    eu_x.append(pt_diff[0])
                    eu_y.append(pt_diff[1])
                    vma.append(ma)
        if not len(vma) > 0:
            set_trace()
            print("unable to find valid match for frame {} and {}".format(
                last_registered, fid))
            drop_idx.append(fid)
            continue
        p_src, p_dst, vma = np.array(p_src), np.array(p_dst), np.array(vma)
        eu_d, eu_x, eu_y = np.array(eu_d), np.array(eu_x), np.array(eu_y)
        if z_th:
            eu_x_z_mask = np.absolute(zscore(eu_x)) < z_th
            eu_y_z_mask = np.absolute(zscore(eu_y)) < z_th
            eu_dist_z_mask = np.absolute(zscore(eu_d)) < z_th
            dist_z_mask = np.logical_and.reduce(
                [eu_dist_z_mask, eu_x_z_mask, eu_y_z_mask])
        else:
            dist_z_mask = np.ones_like(eu_d)
        if q_th:
            x_h_ma = eu_x < np.percentile(eu_x, q_th)
            x_l_ma = eu_x > np.percentile(eu_x, 100 - q_th)
            y_h_ma = eu_y < np.percentile(eu_y, q_th)
            y_l_ma = eu_y > np.percentile(eu_y, 100 - q_th)
            d_h_ma = eu_d < np.percentile(eu_d, q_th)
            d_l_ma = eu_d > np.percentile(eu_d, 100 - q_th)
            dist_q_mask = np.logical_and.reduce(
                [x_h_ma, x_l_ma, y_h_ma, y_l_ma, d_h_ma, d_l_ma])
        else:
            dist_q_mask = np.ones_like(eu_d)
        mask = np.logical_and(dist_z_mask, dist_q_mask)
        p_src, p_dst, vma = p_src[mask], p_dst[mask], vma[mask]
        eu_d, eu_x, eu_y = eu_d[mask], eu_x[mask], eu_y[mask]
        if not len(vma) > 0:
            sys.stdout.write("\033[K")
            print("No matches passed consistency test for frame {} and {}".
                  format(last_registered, fid))
            drop_idx.append(fid)
            continue
        trans, hmask = cv2.findHomography(
            p_src, p_dst, method=cv2.RANSAC, ransacReprojThreshold=1)
        hmask = hmask.squeeze().astype(bool)
        p_src, p_dst, vma = p_src[hmask], p_dst[hmask], vma[hmask]
        eu_d, eu_x, eu_y = eu_d[hmask], eu_x[hmask], eu_y[hmask]
        if not len(vma) > 0:
            sys.stdout.write("\033[K")
            print("no matches formed a homography for frame {} and {}".format(
                last_registered, fid))
            drop_idx.append(fid)
            continue
        elif np.std(eu_d) > std_thres if std_thres else False:
            sys.stdout.write("\033[K")
            print("dist variance too high for frame {} and {}. variance: {}".
                  format(last_registered, fid, np.std(eu_d)))
            drop_idx.append(fid)
            continue
        elif np.std(eu_x) > std_thres if std_thres else False:
            sys.stdout.write("\033[K")
            print("x variance too high for frame {} and {}. variance: {}".
                  format(last_registered, fid, np.std(eu_x)))
            drop_idx.append(fid)
            continue
        elif np.std(eu_y) > std_thres if std_thres else False:
            sys.stdout.write("\033[K")
            print("y variance too high for frame {} and {}. variance: {}".
                  format(last_registered, fid, np.std(eu_y)))
            drop_idx.append(fid)
            continue
        est_shift = np.median(p_dst - p_src, axis=0)
        pts_src = cv2.goodFeaturesToTrack(im_src, 100, 0.5, 3, blockSize=3)
        if pts_src is None or not len(pts_src) > 1:
            sys.stdout.write("\033[K")
            print(
                "not enough good corners for frame {}".format(last_registered))
            drop_idx.append(fid)
            continue
        pts_dst = cv2.goodFeaturesToTrack(im_dst, 100, 0.5, 3, blockSize=3)
        if pts_dst is None or not len(pts_dst) > 1:
            sys.stdout.write("\033[K")
            print("not enough good corners for frame {}".format(fid))
            drop_idx.append(fid)
            continue
        pts_src = np.array(pts_src).squeeze().astype(np.float32)
        pts_dst = pts_src + est_shift
        pts_dst = np.array(pts_dst).astype(np.float32)
        vld_mask = pts_dst.min(axis=1) > 0
        if not vld_mask.sum() > 0:
            sys.stdout.write("\033[K")
            print("no valid corners for frame {} and {}".format(
                last_registered, fid))
            drop_idx.append(fid)
            continue
        pts_src, pts_dst = pts_src[vld_mask], pts_dst[vld_mask]
        p1, st0, err0 = cv2.calcOpticalFlowPyrLK(im_src, im_dst,
                                                 pts_src.copy(),
                                                 pts_dst.copy(), **lk_params)
        p0r, st1, err1 = cv2.calcOpticalFlowPyrLK(im_dst, im_src,
                                                  p1.copy(),
                                                  pts_src.copy(), **lk_params)
        d_mask = np.absolute(pts_src - p0r).reshape(-1, 2).max(-1) < 1
        st0 = st0.squeeze().astype(bool)
        st1 = st1.squeeze().astype(bool)
        optmask = np.logical_and.reduce([st0, st1, d_mask])
        if not np.any(optmask):
            sys.stdout.write("\033[K")
            print(("no valid optical flow matching was found "
                   "for frame {} and {}").format(last_registered, fid))
            drop_idx.append(fid)
            continue
        pts_src, pts_dst, err0 = p0r[optmask], p1[optmask], err0[optmask]
        if err0.mean() > opt_err_thres:
            sys.stdout.write("\033[K")
            print(("optical flow error too high "
                   "for frame {} and {}. error: {}").format(
                       last_registered, fid, err0.mean()))
            drop_idx.append(fid)
            continue
        # consmask = np.absolute(pts_src - pts_dst - est_shift).max(
        #     axis=1) < opt_restr
        # if not consmask.sum() > 0:
        #     print(("no optical flow was found consitent with surf result "
        #            "for frame {} and {}").format(last_registered, fid))
        #     drop_idx.append(fid)
        #     continue
        # pts_src, pts_dst = pts_src[consmask], pts_dst[consmask]
        if len(pts_src) > 3:
            trans, hmask = cv2.findHomography(
                pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=3)
            hmask = hmask.squeeze().astype(bool)
        else:
            hmask = np.ones(len(pts_src), dtype=bool)
        if hmask.sum() < opt_h_prop * len(hmask):
            sys.stdout.write("\033[K")
            print(("too many optical flow matches were outliers "
                   "for frame {} and {}").format(last_registered, fid))
            hmask = np.ones(len(pts_src), dtype=bool)
        pts_src = pts_src[hmask]
        pts_dst = pts_dst[hmask]
        pts_diff = pts_src - pts_dst
        pts_dist = np.sqrt((pts_diff**2).sum(axis=1))
        if np.std(pts_dist) > opt_std_thres:
            sys.stdout.write("\033[K")
            print(("optical flow distance variance too high "
                   "for frame {} and {}. std:{}").format(
                       last_registered, fid, np.std(pts_dist)))
            drop_idx.append(fid)
            continue
        pts_src = pts_src.reshape((-1, 2))
        pts_dst = pts_dst.reshape((-1, 2))
        if method == 'opencv':
            trans = cv2.estimateRigidTransform(pts_dst, pts_src, False)
            if trans is not None:
                varr_mc.loc[dict(frame=fid)] = cv2.warpAffine(
                    varr_mc.sel(frame=fid).values,
                    trans,
                    varr_mc.sel(frame=fid).values.shape[::-1])
            else:
                print("unable to find transform for frame {}".format(fid))
        elif method == 'translation':
            if weight and len(pts_dist) > 1:
                weights = np.exp(-np.array(np.absolute(zscore(pts_dist))) * 10)
                weights = weights / np.sum(weights)
            else:
                weights = None
            shift = estimate_translation(pts_src, pts_dst, weights)
            shifts.append(shift)
            shifts_idx.append(fid)
        elif method == 'skimage':
            trans = tf.estimate_transform('similarity', pts_src, pts_dst)
            varr_mc.loc[dict(frame=fid)] = tf.warp(
                varr_mc.sel(frame=fid), trans.inverse)
        print(
            ("processing frame {:5d} of {:5d}, "
             "current features: {:3d}, current err: {:06.4f}").format(
                 i, len(frm_idx), len(pts_src), err0.mean()),
            end='\r')
        last_registered = fid
        match_dict[fid] = dict(
            src=detect_src,
            dst=detect_dst,
            match=vma,
            src_fid=last_registered,
            upsample=upsample if upsample else 1)
    if method == 'translation':
        shifts = xr.DataArray(
            shifts,
            coords=dict(frame=shifts_idx),
            dims=['frame', 'shift_dims'])
        shifts_final = []
        for fid in frm_idx[1:].values:
            cur_sh_hist = shifts.sel(frame=slice(frm_idx[0], fid))
            cur_shift = cur_sh_hist.sum('frame')
            cur_shift = cur_shift.values.astype(int)
            shifts_final.append(cur_shift)
            varr_mc.loc[dict(frame=fid)] = apply_translation(
                varr_mc.sel(frame=fid), cur_shift)
        shifts_final = xr.DataArray(
            shifts_final,
            coords=dict(frame=frm_idx[1:]),
            dims=['frame', 'shift_dims'])
    else:
        shifts_final = None
    varr_mc = varr_mc.reindex(
        method='nearest',
        width=varray.coords['width'],
        height=varray.coords['height'])
    return varr_mc.rename(varray.name + "_MotionCorrected"
                          ), match_dict, np.array(drop_idx), shifts_final


def remove_duplicate_keypoints(detect, threshold=2):
    remv_idx = []
    kps = detect[0]
    des = detect[1]
    for kp0, kp1 in itt.combinations(enumerate(kps), 2):
        if not (kp0[0] in remv_idx or kp1[0] in remv_idx):
            dist = np.sqrt(
                np.sum(np.array(kp0[1].pt) - np.array(kp1[1].pt))**2)
            if dist < threshold:
                remv_idx.append(kp0[0])
    kps = [kp for ikp, kp in enumerate(kps) if ikp not in remv_idx]
    des = np.delete(des, remv_idx, axis=0)
    return (kps, des)


def estimate_translation(pts_src, pts_dst, weights=None):
    return np.average(pts_src - pts_dst, axis=0, weights=weights)
    # return np.median(pts_src - pts_dst, axis=0)


def apply_translation(img, shift):
    return np.roll(img, shift, axis=(1, 0))


def mask_shifts(varr_fft, corr, shifts, z_thres, perframe=True, pad_f=1):
    dims = list(varr_fft.dims)
    dims.remove('frame')
    sizes = [varr_fft.sizes[d] for d in dims]
    pad_s = np.array(sizes) * pad_f
    pad_s = pad_s.astype(int)
    mask = xr.apply_ufunc(zscore, corr.fillna(0)) > z_thres
    shifts = shifts.where(mask)
    if perframe:
        mask_diff = xr.DataArray(
            np.diff(mask.astype(int)),
            coords=dict(frame=mask.coords['frame'][1:]),
            dims=['frame'])
        good_idx = mask.coords['frame'].where(mask > 0, drop=True)
        bad_idx = mask_diff.coords['frame'].where(mask_diff == -1, drop=True)
        for cur_bad in bad_idx:
            gb_diff = good_idx - cur_bad
            try:
                next_good = gb_diff[gb_diff > 0].min() + cur_bad
                last_good = gb_diff[gb_diff < 0].max() + cur_bad
                cur_src = varr_fft.sel(frame=last_good)
                cur_dst = varr_fft.sel(frame=next_good)
                res = match_temp(cur_src, cur_dst, pad_s, pad_f)
                shifts.loc[dict(frame=next_good.values)] = res
            except (KeyError, ValueError):
                print("unable to correct for bad frame: {}".format(int(cur_bad)))
    return shifts, mask


def estimate_shifts(varr, max_sh, dim="frame", npart=3, local=False):
    varr = varr.chunk({"height": -1, "width": -1})
    vmax, sh = xr.apply_ufunc(
        est_sh_part,
        varr,
        input_core_dims=[[dim, "height", "width"]],
        output_core_dims=[["height", "width"], [dim, "variable"]],
        dask="allowed",
        kwargs={"max_sh": max_sh, "npart": npart, "local": local},
    )
    return sh.assign_coords(variable=["height", "width"])


def est_sh_part(varr, max_sh, npart, local):
    if varr.shape[0] <= 1:
        return varr.squeeze(), np.array([[0, 0]])
    idx_spt = np.array_split(np.arange(varr.shape[0]), npart)
    fm_ls, sh_ls = [], []
    for idx in idx_spt:
        if len(idx) > 0:
            fm, sh = est_sh_part(varr[idx, :, :], max_sh, npart, local)
            fm_ls.append(fm)
            sh_ls.append(sh)
    mid = int(len(sh_ls) / 2)
    sh_add_ls = [np.array([0, 0])] * len(sh_ls)
    for i, fm in enumerate(fm_ls):
        if i < mid:
            temp = fm_ls[i + 1]
            sh_idx = np.arange(i + 1)
        elif i > mid:
            temp = fm_ls[i - 1]
            sh_idx = np.arange(i, len(sh_ls))
        else:
            continue
        sh_add = darr.from_delayed(
            delayed(match_temp)(fm, temp, max_sh, local), (2,), float
        )
        for j in sh_idx:
            sh_ls[j] = sh_ls[j] + sh_add.reshape((1, -1))
            sh_add_ls[j] = sh_add_ls[j] + sh_add
    for i, (fm, sh) in enumerate(zip(fm_ls, sh_add_ls)):
        fm_ls[i] = darr.nan_to_num(
            darr.from_delayed(delayed(shift_perframe)(fm, sh), fm.shape, fm.dtype)
        )
    sh_ret = darr.concatenate(sh_ls)
    fm_ret = darr.stack(fm_ls)
    return fm_ret.max(axis=0), sh_ret


def match_temp(src, dst, max_sh, local, subpixel=False):
    src = np.pad(src, max_sh)
    cor = cv2.matchTemplate(
        src.astype(np.float32), dst.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    if not len(np.unique(cor)) > 1:
        return np.array([0, 0])
    cent = np.floor(np.array(cor.shape) / 2)
    if local:
        cor_ma = cv2.dilate(cor, np.ones((3, 3)))
        maxs = np.array(np.nonzero(cor_ma == cor))
        dev = ((maxs - cent[:, np.newaxis]) ** 2).sum(axis=0)
        imax = maxs[:, np.argmin(dev)]
    else:
        imax = np.unravel_index(np.argmax(cor), cor.shape)
    if subpixel:
        x0 = np.arange(max(imax[0] - 5, 0), min(imax[0] + 6, cor.shape[0]))
        x1 = np.arange(max(imax[1] - 5, 0), min(imax[1] + 6, cor.shape[1]))
        y0 = cor[x0, imax[1]]
        y1 = cor[imax[0], x1]
        p0 = np.polyfit(x0, y0, 2)
        p1 = np.polyfit(x1, y1, 2)
        imax = np.array([
            -0.5 * p0[1] / p0[0],
            -0.5 * p1[1] / p1[0]
        ])
        # m0 = np.roots(np.polyder(p0))
        # m1 = np.roots(np.polyder(p1))
        # m0 = m0[np.argmin(np.abs(m0 - imax[0]))]
        # m1 = m1[np.argmin(np.abs(m1 - imax[1]))]
        # imax = np.array([m0, m1])
    sh = cent - imax
    return sh


def apply_shifts(varr, shifts):
    sh_dim = shifts.coords['variable'].values.tolist()
    varr_sh = xr.apply_ufunc(
        shift_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        shifts,
        input_core_dims=[sh_dim, ['variable']],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask = 'parallelized',
        output_dtypes = [varr.dtype])
    return varr_sh


def mser_thres(fm):
    mser = cv2.MSER_create(_min_area=100, _delta=1, _max_variation=2)
    reg, _ = mser.detectRegions(fm.T)
    reg_set = [set(tuple(crd) for crd in r) for r in reg]
    fm_thres = np.zeros_like(fm)
    if reg_set:
        reg = np.array(list(set.union(*reg_set)))
        fm_thres[reg[:, 0], reg[:, 1]] = fm[reg[:, 0], reg[:, 1]]
    return fm_thres


def shift_perframe(fm, sh):
    sh = np.around(sh).astype(int)
    fm = np.roll(fm, sh, axis=np.arange(fm.ndim))
    index = [slice(None) for _ in range(fm.ndim)]
    for ish, s in enumerate(sh):
        index = [slice(None) for _ in range(fm.ndim)]
        if s > 0:
            index[ish] = slice(None, s)
            fm[tuple(index)] = np.nan
        elif s == 0:
            continue
        elif s < 0:
            index[ish] = slice(s, None)
            fm[tuple(index)] = np.nan
    return fm


def interpolate_frame(varr, drop_idx):
    if drop_idx.dtype == bool:
        drop_idx = drop_idx.coords['frame'].where(~drop_idx, drop=True).values
    if not set(drop_idx):
        print("no bad frame to interpolate, returning input")
        return varr
    keep_idx = np.array(list(set(varr.coords['frame'].values) - set(drop_idx)))
    varr_int = varr.copy()
    for i, fid in enumerate(drop_idx):
        print(
            "processing frame: {} progress: {}/{}".format(
                fid, i, len(drop_idx)),
            end='\r')
        diff = keep_idx - fid
        try:
            fid_fwd = diff[diff > 0].min() + fid
        except ValueError:
            fid_fwd = keep_idx.max()
        try:
            fid_bak = diff[diff < 0].max() + fid
        except ValueError:
            fid_bak = keep_idx.min()
        int_src = xr.concat(
            [varr.sel(frame=fid_fwd),
             varr.sel(frame=fid_bak)], dim='frame')
        varr_int.loc[dict(frame=fid)] = int_src.mean('frame')
    print("\ninterpolation done")
    return varr_int.rename(varr.name + "_Interpolated")
