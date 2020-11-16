import itertools as itt
import os

import cv2
import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import xarray as xr
from dask.delayed import delayed
from scipy.ndimage.filters import median_filter
from scipy.ndimage.measurements import label
from scipy.signal import butter, lfilter
from scipy.stats import kstest, zscore
from skimage.morphology import disk
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

from .cnmf import label_connected, smooth_sig
from .utilities import save_minian


def seeds_init(
    varr,
    wnd_size=500,
    method="rolling",
    stp_size=200,
    nchunk=100,
    max_wnd=10,
    diff_thres=2,
):
    """
    This function computes the maximum intensity projection of a subset of frames and finds the local maxima. The set of local maxima constitutes  an overly-complete set of local maxima, which are the putative locations of cells, which we call seeds.

    Args:
        varr (xarray.DataArray): input data
        wnd_size (int, optional): size of the temporal window in which the maximum intensity projection will be computed, i.e. number of frames. Defaults to 500.
        method (str, optional): proceeds through the data in temporal order, alternative is randomly. Defaults to 'rolling'.
        stp_size (int, optional): only if the method is rolling, defines the step size. Defaults to 200.
        nchunk (int, optional): only if the method is random, defines the number of chunks randomly picked. Defaults to 100.
        max_wnd (int, optional): max size (in pixel) of the diameter for cell detection. Defaults to 10.
        diff_thres (int, optional): minimal fluorescence difference of a seed across frames. Defaults to 2.

    Returns:
        pandas.core.frame.DataFrame: matrix of seeds
    """
    int_path = os.environ["MINIAN_INTERMEDIATE"]
    print("constructing chunks")
    idx_fm = varr.coords["frame"]
    nfm = len(idx_fm)
    if method == "rolling":
        nstp = np.ceil(nfm / stp_size) + 1
        centers = np.linspace(0, nfm - 1, int(nstp))
        hwnd = np.ceil(wnd_size / 2)
        max_idx = list(
            map(
                lambda c: slice(
                    int(np.floor(c - hwnd).clip(0)), int(np.ceil(c + hwnd))
                ),
                centers,
            )
        )
    elif method == "random":
        max_idx = [np.random.randint(0, nfm - 1, wnd_size) for _ in range(nchunk)]
    print("computing max projections")
    res = [max_proj_frame(varr, cur_idx) for cur_idx in max_idx]
    max_res = xr.concat(res, "sample")
    max_res = save_minian(max_res.rename("max_res"), int_path, overwrite=True)
    print("calculating local maximum")
    loc_max = xr.apply_ufunc(
        local_max_roll,
        max_res,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.uint8],
        kwargs=dict(k0=2, k1=max_wnd, diff=diff_thres),
    ).sum("sample")
    seeds = (
        loc_max.where(loc_max > 0).rename("seeds").to_dataframe().dropna().reset_index()
    )
    return seeds[["height", "width", "seeds"]]


def max_proj_frame(varr, idx):
    return varr.isel(frame=idx).max("frame")


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


def gmm_refine(
    varr, seeds, q=(0.1, 99.9), n_components=2, valid_components=1, mean_mask=True
):
    """
    Estimate the initial parameters to estimate the distribution of fluorescence intensity using Gaussian mixture model

    Args:
        varr (xarray.DataArray): input data
        seeds (dict): seeds value
        q (tuple, optional): the quantile of signal of each seed, from which the peak-to-peak values are calculated. i.e., a value of (0,1) will be equivalent to defining the peak-to-peak value as the difference between minimum and maximum. However it’s usually useful to not use the absolute minimum and maximum so that the algorithm is more resilient to outliers. Defaults to (0.1, 99.9).
        n_components (int, optional): number of mixture components. Defaults to 2.
        valid_components (int, optional): number of mixture components to be considered signal. Defaults to 1.
        mean_mask (bool, optional): whether to apply additional criteria where a seed is valid only if its peak-to-peak value exceeds the mean of the lowest gaussian distribution, only useful in corner cases where the distribution of the gaussian heavily overlap. Defaults to True.

    Returns:
        [dict]: [seeds, signal range]
    """
    print("selecting seeds")
    varr_sub = varr.sel(spatial=[tuple(hw) for hw in seeds[["height", "width"]].values])
    print("computing peak-valley values")
    varr_valley = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        kwargs=dict(q=q[0], axis=-1),
        dask="parallelized",
        output_dtypes=[varr_sub.dtype],
    )
    varr_peak = xr.apply_ufunc(
        np.percentile,
        varr_sub.chunk(dict(frame=-1)),
        input_core_dims=[["frame"]],
        kwargs=dict(q=q[1], axis=-1),
        dask="parallelized",
        output_dtypes=[varr_sub.dtype],
    )
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
    seeds["mask_gmm"] = idx_valid
    return seeds, varr_pv, gmm


def pnr_refine(varr, seeds, noise_freq=0.25, thres=1.5, q=(0.1, 99.9), med_wnd=None):
    """
    "peak-to-noise ratio" refine. This function computes the ratio between noise range and the signal range, where signal is defined as the lower half of the frequency range, while noise is the higher half of the frequency range.

    Args:
        varr (xarray.DataArray): data array
        seeds (pandas.core.frame.DataFrame): seeds
        noise_freq (float, optional): frequency of the noise. Defaults to 0.25.
        thres (float, optional): threshold. Defaults to 1.5.
        q (tuple, optional): Defaults to (0.1, 99.9).
        med_wnd (type, optional): if specified, a median filter with the set window size is applied to the signal from each seeds and subtracted from signal. Useful if there’s a shift in baseline fluorescence that produce lots of false positive seeds. Defaults to None.

    Returns:
        [tuple pandas.core.frame.DataFrame, scikit-learn gmm object]: seeds, peak to noise ratio and gaussian mixture model.

    """
    print("selecting seeds")
    varr_sub = xr.concat(
        [varr.sel(height=h, width=w) for h, w in seeds[["height", "width"]].values],
        "index",
    ).assign_coords({"index": seeds.index.values})
    if med_wnd:
        print("removing baseline")
        varr = xr.apply_ufunc(
            med_baseline,
            varr_sub,
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            kwargs={"wnd": med_wnd},
            vectorize=True,
            output_dtypes=[varr.dtype],
        )
    print("computing peak-noise ratio")
    pnr = xr.apply_ufunc(
        pnr_perseed,
        varr_sub,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        kwargs={"freq": noise_freq, "q": q},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()
    if thres == "auto":
        gmm = GaussianMixture(n_components=2)
        gmm.fit(np.nan_to_num(pnr.values.reshape(-1, 1)))
        idg = np.argsort(gmm.means_.reshape(-1))[-1]
        idx_valid = np.isin(gmm.predict(pnr.values.reshape(-1, 1)), idg)
        seeds["mask_pnr"] = idx_valid
    else:
        mask = pnr > thres
        mask_df = mask.to_pandas().rename("mask_pnr")
        seeds["mask_pnr"] = mask_df
        gmm = None
    return seeds, pnr, gmm


def ptp_q(a, q):
    return np.percentile(a, q[1]) - np.percentile(a, q[0])


def pnr_perseed(a, freq, q):
    ptp = ptp_q(a, q)
    but_b, but_a = butter(2, freq, btype="high", analog=False)
    a = lfilter(but_b, but_a, a).real
    ptp_noise = ptp_q(a, q)
    return ptp / ptp_noise


def med_baseline(a, wnd):
    base = median_filter(a, size=wnd)
    a -= base
    return a


def intensity_refine(varr, seeds, thres_mul=2):
    try:
        fm_max = varr.max("frame")
    except ValueError:
        print("using input as max projection")
        fm_max = varr
    bins = np.around(fm_max.sizes["height"] * fm_max.sizes["width"] / 10).astype(int)
    hist, edges = np.histogram(fm_max, bins=bins)
    try:
        thres = edges[int(np.around(np.argmax(hist) * thres_mul))]
    except IndexError:
        print("threshold out of bound, returning input")
        return seeds
    mask = (fm_max > thres).stack(spatial=["height", "width"])
    mask_df = mask.to_pandas().rename("mask_int").reset_index()
    seeds = pd.merge(seeds, mask_df, on=["height", "width"], how="left")
    return seeds


def ks_refine(varr, seeds, sig=0.01):
    """
    This function refines the seeds using Kolmogorov-Smirnov (KS) test. This step is based on the assumption that the seeds’ fluorescence across frames notionally follows a bimodal distribution: with a large normal distribution representing baseline activity, and a second peak representing when the seed/cell is active. KS allows to discard the seeds where the null-hypothesis (i.e. the fluorescence intensity is simply a normal distribution) is rejected ad alpha = 0.05.

    Args:
        varr (xarray.DataArray): flattened version of the video
        seeds (dict): seeds
        sig (float, optional): alpha. Defaults to 0.05.

    Returns:
        dict: seeds
    """
    print("selecting seeds")
    varr_sub = xr.concat(
        [varr.sel(height=h, width=w) for h, w in seeds[["height", "width"]].values],
        "index",
    ).assign_coords({"index": seeds.index.values})
    print("performing KS test")
    ks = xr.apply_ufunc(
        ks_perseed,
        varr_sub,
        input_core_dims=[["frame"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    ).compute()
    ks = (ks < sig).to_pandas().rename("mask_ks")
    seeds["mask_ks"] = ks
    return seeds


def ks_perseed(a):
    a = zscore(a)
    return kstest(a, "norm")[1]


def seeds_merge(varr, max_proj, seeds, thres_dist=5, thres_corr=0.6, noise_freq=None):
    """
    This function merges neighboring seeds which potentially come from the same cell, based upon their spatial distance and temporal correlation of their activity

    Args:
        varr (xarray.DataArray): input data
        seeds (dict): seeds
        thres_dist (int, optional): spatial distance threshold. Defaults to 5.
        thres_corr (float, optional): activity correlation threshold. Defaults to 0.6.
        noise_freq (str, optional): noise frequency. Defaults to 'envelope'.

    Returns:
        dict: seeds
    """
    if noise_freq:
        varr = smooth_sig(varr, noise_freq)
    print("computing distance")
    dist = kneighbors_graph(seeds[["height", "width"]], n_neighbors=1, mode="distance")
    print("computing correlations")
    corr_ls = []
    row_idx = []
    col_idx = []
    varr = varr - varr.mean("frame")
    std = np.sqrt((varr ** 2).sum("frame"))
    for i, j in zip(*dist.nonzero()):
        if dist[i, j] < thres_dist:
            hi, hj, wi, wj = (
                seeds.iloc[i]["height"],
                seeds.iloc[j]["height"],
                seeds.iloc[i]["width"],
                seeds.iloc[j]["width"],
            )
            varr_i, varr_j, std_i, std_j = (
                varr.sel(height=hi, width=wi),
                varr.sel(height=hj, width=wj),
                std.sel(height=hi, width=wi),
                std.sel(height=hj, width=wj),
            )
            corr = (varr_i * varr_j).sum() / (std_i * std_j)
            corr_ls.append(corr)
            row_idx.append(i)
            col_idx.append(j)
    corr_ls = dask.compute(corr_ls)[0]
    print("merging seeds")
    adj = (
        scipy.sparse.csr_matrix((corr_ls, (row_idx, col_idx)), shape=dist.shape)
        > thres_corr
    )
    adj = adj + adj.T
    labels = label_connected(adj, only_connected=True)
    iso = np.where(labels < 0)[0]
    seeds_final = set(iso.tolist())
    for cur_cmp in np.unique(labels):
        if cur_cmp < 0:
            continue
        cur_smp = np.where(labels == cur_cmp)[0]
        cur_max = np.array(
            [
                max_proj.sel(
                    height=seeds.iloc[s]["height"], width=seeds.iloc[s]["width"]
                )
                for s in cur_smp
            ]
        )
        max_seed = cur_smp[np.argmax(cur_max)]
        seeds_final.add(max_seed)
    seeds["mask_mrg"] = False
    seeds.loc[list(seeds_final), "mask_mrg"] = True
    return seeds


def initA(varr, seeds, thres_corr=0.8, wnd=10, noise_freq=None):
    seeds = seeds.sort_values(["height", "width"])
    if noise_freq:
        print("smoothing signal")
        varr = smooth_sig(varr, noise_freq)
    print("computing correlations")
    varr = varr - varr.mean("frame")
    std = np.sqrt((varr ** 2).sum("frame"))
    res_ls = [
        initA_perseed(varr, std, h, w, wnd, thres_corr)
        for h, w in zip(seeds["height"].values, seeds["width"].values)
    ]
    A_ls = []
    for i in range(0, len(res_ls), 50):
        cur_ls = dask.compute(res_ls[i : i + 50])[0]
        A_ls.extend([a.chunk() for a in cur_ls])
    A = xr.concat(A_ls, "unit_id")
    A = A.assign_coords(unit_id=np.arange(A.sizes["unit_id"]))
    A.data = A.data.map_blocks(lambda a: a.todense(), dtype=float)
    return A


def initA_perseed(varr, std, h, w, wnd, thres_corr):
    ih = np.where(varr.coords["height"] == h)[0][0]
    iw = np.where(varr.coords["width"] == w)[0][0]
    h_sur, w_sur = (
        np.arange(max(ih - wnd, 0), min(ih + wnd, varr.sizes["height"])),
        np.arange(max(iw - wnd, 0), min(iw + wnd, varr.sizes["width"])),
    )
    sur = varr.isel(height=h_sur, width=w_sur)
    corr = (
        varr.isel(height=ih, width=iw).dot(sur) / std.isel(height=ih, width=iw) / std
    ).data.reshape(-1)
    corr = da.where(corr > thres_corr, corr, 0)
    crds = np.array(list(itt.product(h_sur, w_sur))).T
    corr = delayed(sparse.COO)(
        crds,
        corr,
        shape=(varr.sizes["height"], varr.sizes["width"]),
    )
    corr = da.from_delayed(
        corr, shape=(varr.sizes["height"], varr.sizes["width"]), dtype=float
    )
    return xr.DataArray(
        corr,
        dims=["height", "width"],
        coords={
            "height": varr.coords["height"].values,
            "width": varr.coords["width"].values,
        },
    )


def initC(varr, A):
    uids = A.coords["unit_id"]
    fms = varr.coords["frame"]
    A = A.data.map_blocks(sparse.COO).map_blocks(lambda a: a / a.sum()).rechunk(-1)
    C = da.tensordot(A, varr, axes=[(1, 2), (1, 2)])
    C = xr.DataArray(
        C, dims=["unit_id", "frame"], coords={"unit_id": uids, "frame": fms}
    )
    return C


def initbf(varr, A, C):
    A = A.data.map_blocks(sparse.COO).compute()
    Yb = (varr - da.tensordot(C, A, axes=[(0,), (0,)])).clip(0)
    intpath = os.environ["MINIAN_INTERMEDIATE"]
    Yb = save_minian(Yb.rename("Yb"), intpath, overwrite=True)
    b = Yb.mean("frame")
    f = Yb.mean(["height", "width"])
    return b, f


@da.as_gufunc(signature="(h, w)->(h, w)", output_dtypes=int, allow_rechunk=True)
def da_label(im):
    """[summary]

    Args:
        im ([type]): [description]

    Returns:
        [type]: [description]
    """
    return label(im)[0]
