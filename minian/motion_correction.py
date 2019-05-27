import numpy as np
import xarray as xr
import cv2
import sys
import itertools as itt
import pyfftw.interfaces.numpy_fft as npfft
import numba as nb
import dask.array as darr
import pandas as pd
# import affine6p
import SimpleITK as sitk
from minian.cnmf import label_connected
from scipy.stats import zscore
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
from skimage import transform as tf
from scipy.stats import zscore
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from IPython.core.debugger import set_trace


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


def mser_vec(mov):
    if len(mov.shape) == 3:
        res_ls = [mser(fm) for fm in mov]
        return res_ls
    elif len(mov.shape) == 2:
        return [mser(mov)]
    else:
        raise ValueError("movie shape {} not supported".format(mov.shape))


def mser(fm, merge=True):
    mser = cv2.MSER_create(_delta=3, _max_variation=0.5, _max_area=400)
    mser.setPass2Only(True)
    reg, _ = mser.detectRegions(fm.T)
    if merge:
        reg = merge_regs(fm, reg)
    return reg

@nb.jit(nopython=True, nogil=True)
def jac(l0, l1):
    s0 = set([tuple(i) for i in l0[0]])
    s1 = set([tuple(i) for i in l1[0]])
    return len(s0.intersection(s1)) / len(s0.union(s1))

def merge_regs(fm, regs, thres=3):
    cnts = reg_to_cnts(fm, regs, unique=False)
    labs = label_connected(squareform(pdist(cnts)) < thres)
    reg_mrg = _mrg(labs, regs)
    return reg_mrg

# @nb.jit(nopython=True, nogil=True)
def _mrg(labs, regs):
    reg_mrg = []
    for lb in np.unique(labs):
        idxs = np.where(labs == lb)[0]
        rg = [tuple(cd) for i in idxs for cd in regs[i]]
        reg_mrg.append(np.stack(list(set(rg)), axis=0))
    return reg_mrg


def mser_mask(fm):
    mser = cv2.MSER_create(_delta=3, _max_variation=0.25, _max_area=400)
    mser.setPass2Only(True)
    reg, _ = mser.detectRegions(fm)
    reg = np.unique(np.concatenate(reg, axis=0), axis=0)
    ma = np.zeros(fm.shape).astype(bool)
    ma[reg[:, 1], reg[:, 0]] = True
    return ma

def desc_vec(mov):
    if len(mov.shape) == 3:
        res_ls = [kaze(fm, mser(fm)) for fm in mov]
        return res_ls
    elif len(mov.shape) == 2:
        return [kaze(fm, mser(mov))]
    else:
        raise ValueError("movie shape {} not supported".format(mov.shape))

def reg_to_cnts(fm, regs, unique=True, weighted=True):
    cnts = []
    for r in regs:
        if weighted:
            w = fm[r[:, 0], r[:, 1]]
        else:
            w = None
        cnt = np.average(r, axis=0, weights=w)
        cnts.append(cnt)
    cnts = np.stack(cnts, axis=0)
    if unique:
        cnts = np.around(cnts).astype(int)
        cnts = np.unique(cnts, axis=0)
    return cnts


def kaze(fm, reg):
    cnts = reg_to_cnts(fm, reg, unique=False, weighted=False)
    kps = [cv2.KeyPoint(x, y, _size=3, _class_id=1) for y, x in cnts]
    kaze = cv2.KAZE_create()
    _, dsc = kaze.compute(fm, kps)
    return cnts, dsc


def desc_rad(fm, reg, wnd=15, ang_bins=8, dist_bins=5):
    cnts = np.around(reg_to_cnts(fm, reg)).astype(int)
    ucnt = np.unique(cnts, axis=0)
    l0, h0, l1, h1 = ucnt[:, 0]-wnd, ucnt[:, 0]+wnd+1, ucnt[:, 1]-wnd, ucnt[:, 1]+wnd+1
    fm_sub = np.stack([padded_slice(fm, (s0, s1, s2, s3)) for s0, s1, s2, s3 in zip(l0, h0, l1, h1)])
    grd = np.stack([vec(sub) for sub in fm_sub], axis=0)
    bins_a, dist = gen_ang_bins(wnd, ang_bins)
    _, bins_d = np.histogram(dist)
    dsc_ls = []
    ma_ls = []
    for iang, idist in itt.product(range(ang_bins), range(dist_bins)):
        d_ma = np.logical_and(dist >= bins_d[idist], dist <= bins_d[idist+1])
        ma = np.logical_and(d_ma, bins_a[iang])
        ma_ls.append(ma)
        ma_crd = np.nonzero(ma)
        cur_dsc = np.nanmean(grd[:, ma_crd[0], ma_crd[1], :], axis=1)
        dsc_ls.append(cur_dsc)
    ma = np.stack(ma_ls, axis=0)
    dsc = np.concatenate(dsc_ls, axis=1)
    return dsc, grd, fm_sub, ucnt, ma


def padded_slice(img, sl):
    output_shape = np.asarray(img.shape)
    output_shape[0] = sl[1] - sl[0]
    output_shape[1] = sl[3] - sl[2]
    src = [max(sl[0], 0),
           min(sl[1], img.shape[0]),
           max(sl[2], 0),
           min(sl[3], img.shape[1])]
    dst = [src[0] - sl[0], src[1] - sl[0],
           src[2] - sl[2], src[3] - sl[2]]
    output = np.zeros(output_shape, dtype=img.dtype)
    output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
    return output


def crd_to_mask(crd, shape):
    ma = np.zeros(shape)
    ma[crd[:, 1], crd[:, 0]] = 1
    return ma

def com(im):
    return center_of_mass(im, (~np.isnan(im)).astype(int))


def vec(im, norm_mag=False):
    gx = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3).astype(float)
    gy = cv2.Sobel(im, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3).astype(float)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if norm_mag:
        mag = (mag - mag.min()) / (mag.max() - mag.min()) * 2 * np.pi
    ang = np.arctan2(gy, gx)
    return np.stack((ang, mag), axis=-1)


def gen_ang_bins(r, ang_bins):
    d = 2 * r + 1
    xs, ys = np.meshgrid(np.arange(d), np.arange(d))
    xs, ys = xs - r, ys - r
    ang = np.arctan2(ys, xs) + np.pi
    bin_sz = 2 * np.pi / ang_bins
    bin_ls = []
    for i in range(ang_bins):
        cur_ma = np.logical_and(ang >= i * bin_sz, ang <= (i + 1) * bin_sz)
        cur_ma[r, r] = False
        bin_ls.append(cur_ma)
    bin_ang = np.stack(bin_ls, axis=0)
    dist = np.sqrt(xs ** 2 + ys ** 2)
    return bin_ang, dist


def desc_vec_rad(mov):
    if len(mov.shape) == 3:
        res_ls = [desc_rad(fm, mser(fm)) for fm in mov]
        return res_ls
    elif len(mov.shape) == 2:
        return [desc_rad(mov, mser(mov))]
    else:
        raise ValueError("movie shape {} not supported".format(mov.shape))


def match_dsc(dsc_src, dsc_dst):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    mas = bf.match(dsc_src.astype(np.float32), dsc_dst.astype(np.float32))
    return mas

def estimate_homo(cnt_src, dsc_src, cnt_dst, dsc_dst):
    mas = match_dsc(dsc_src, dsc_dst)
    ma_df = pd.DataFrame({
        'idx_src': [m.queryIdx for m in mas],
        'idx_dst': [m.trainIdx for m in mas]})
    ma_df['x_src'] = cnt_src[ma_df['idx_src'], 1]
    ma_df['y_src'] = cnt_src[ma_df['idx_src'], 0]
    ma_df['x_dst'] = cnt_dst[ma_df['idx_dst'], 1]
    ma_df['y_dst'] = cnt_dst[ma_df['idx_dst'], 0]
    h, mah = cv2.findHomography(
        ma_df[['x_src', 'y_src']].values, ma_df[['x_dst', 'y_dst']].values,
        method=cv2.LMEDS)
    mah = mah.squeeze().astype(bool)
    ma_df = ma_df[mah]
    aff = np.array(affine6p.estimate(
        ma_df[['x_src', 'y_src']].values, ma_df[['x_dst', 'y_dst']].values)
                   .get_matrix())
    return aff



def estimate_translation(pts_src, pts_dst, weights=None):
    return np.average(pts_src - pts_dst, axis=0, weights=weights)
    # return np.median(pts_src - pts_dst, axis=0)


def apply_translation(img, shift):
    return np.roll(img, shift, axis=(1, 0))


def estimate_shift_fft(varr, dim='frame', on='first', pad_f=1, pct_thres=None):
    varr = varr.chunk(dict(height=-1, width=-1))
    dims = list(varr.dims)
    dims.remove(dim)
    sizes = [varr.sizes[d] for d in ['height', 'width']]
    if not pct_thres:
        pct_thres = (1 - 10 / (sizes[0] * sizes[1])) * 100
    print(pct_thres)
    pad_s = np.array(sizes) * pad_f
    pad_s = pad_s.astype(int)
    results = []
    print("computing fft on array")
    varr_fft = xr.apply_ufunc(
        darr.fft.fft2,
        varr,
        input_core_dims=[[dim, 'height', 'width']],
        output_core_dims=[[dim, 'height', 'width']],
        dask='allowed',
        kwargs=dict(s=pad_s),
        output_dtypes=[np.complex64])
    if on == 'mean':
        meanfm = varr.mean(dim)
        src_fft = xr.apply_ufunc(
            darr.fft.fft2,
            meanfm,
            input_core_dims=[['height', 'width']],
            output_core_dims=[['height', 'width']],
            dask='allowed',
            kwargs=dict(s=pad_s),
            output_dtypes=[np.complex64])
    elif on == 'first':
        src_fft = varr_fft.isel(**{dim: 0})
    elif on == 'last':
        src_fft = varr_fft.isel(**{dim: -1})
    elif on == 'perframe':
        src_fft = varr_fft.shift(**{dim: 1})
    else:
        try:
            src_fft = varr_fft.isel(**{dim: on})
        except TypeError:
            print("template not understood. returning")
            return
    print("estimating shifts")
    res = xr.apply_ufunc(
        shift_fft,
        src_fft,
        varr_fft,
        input_core_dims=[['height', 'width'], ['height', 'width']],
        output_core_dims=[['variable']],
        kwargs=dict(pct_thres=pct_thres),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        output_sizes=dict(variable=3))
    res = res.assign_coords(variable=['height', 'width', 'corr'])
    return res


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
                res = shift_fft(cur_src, cur_dst, pad_s, pad_f)
                shifts.loc[dict(frame=next_good.values)] = res[0:2]
            except (KeyError, ValueError):
                print("unable to correct for bad frame: {}".format(int(cur_bad)))
    return shifts, mask


def shift_fft(fft_src, fft_dst, pad_s=None, pad_f=1, pct_thres=99.99):
    if not np.iscomplexobj(fft_src):
        fft_src = np.fft.fft2(fft_src)
    if not np.iscomplexobj(fft_dst):
        fft_dst = np.fft.fft2(fft_dst)
    if np.isnan(fft_src).any() or np.isnan(fft_dst).any():
        return np.array([0, 0, np.nan])
    dims = fft_dst.shape
    prod = fft_src * np.conj(fft_dst)
    iprod = np.fft.ifft2(prod)
    iprod_sh = np.fft.fftshift(iprod)
    cor = iprod_sh.real
    # cor = np.log(np.where(iprod_sh.real > 1, iprod_sh.real, 1))
    cor_cent = np.where(cor > np.percentile(cor, pct_thres), cor, 0)
    sh = center_of_mass(cor_cent) - np.ceil(np.array(dims) / 2.0 * pad_f)
    # sh = np.unravel_index(cor.argmax(), cor.shape) - np.ceil(np.array(dims) / 2.0 * pad_f)
    corr = np.max(iprod.real)
    return np.concatenate([sh, corr], axis=None)


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


def demon(src, dst, ma_src=None, ma_dst=None, niter=10, std=1, stp=10):
    if ma_src is not None:
        src = np.where(ma_src, src, 0)
    if ma_dst is not None:
        dst = np.where(ma_dst, dst, 0)
    im_src = sitk.GetImageFromArray(src)
    im_dst = sitk.GetImageFromArray(dst)
    dem = sitk.DiffeomorphicDemonsRegistrationFilter()
    # dem = sitk.DemonsRegistrationFilter()
    # dem = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    dem.SetNumberOfIterations(niter)
    dem.SetStandardDeviations(std)
    # dem.SetMaximumUpdateStepLength(stp)
    # dem.SetUseGradientType(dem.Fixed)
    trans = dem.Execute(im_dst, im_src)
    return sitk.DisplacementFieldTransform(trans)

def demon_reg(src, dst, src_regs, dst_regs):
    src_cnts = reg_to_cnts(src, src_regs, unique=False)
    dst_cnts = reg_to_cnts(dst, dst_regs, unique=False)
    mas = match_dsc(src_cnts, dst_cnts)
    src_mch = np.zeros_like(src)
    dst_mch = np.zeros_like(dst)
    for im, m in enumerate(mas):
        id_src = src_regs[m.queryIdx]
        id_dst = dst_regs[m.trainIdx]
        cur_src = src[id_src[:, 0], id_src[:, 1]]
        cur_dst = dst[id_dst[:, 0], id_dst[:, 1]]
        src_ma = hist_match(cur_src, cur_dst)
        src_mch[id_src[:, 0], id_src[:, 1]] = src_ma
        dst_mch[id_dst[:, 0], id_dst[:, 1]] = cur_dst
    dis = demon(src_mch, dst_mch)
    return dis


def apply_displacement(dis, im):
    if dis is None:
        return im
    im = sitk.GetImageFromArray(im)
    resamp = sitk.ResampleImageFilter()
    resamp.SetReferenceImage(im)
    resamp.SetInterpolator(sitk.sitkLinear)
    resamp.SetTransform(dis)
    im_out = resamp.Execute(im)
    return sitk.GetArrayFromImage(im_out)

# @nb.jit(nopython=True, nogil=True)
def hist_match(src, dst, nbins=128):
    mmax = max(src.max(), dst.max())
    mmin = min(src.min(), dst.min())
    bins = np.linspace(mmin, mmax, nbins)
    src_his, _ = np.histogram(src, bins, density=True)
    dst_his, _ = np.histogram(dst, bins, density=True)
    src_cdf = src_his.cumsum()
    dst_cdf = dst_his.cumsum()
    cdf_pix = np.interp(src, bins[:-1], src_cdf)
    src_map = np.interp(cdf_pix, dst_cdf, bins[:-1])
    return src_map


def lin_match(src, dst):
    smin, smax = src.min(), src.max()
    dmin, dmax = dst.min(), dst.max()
    src_ma = (src - smin)/(smax - smin)
    src_ma = src_ma * (dmax - dmin) + dmin
    return src_ma
