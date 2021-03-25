import functools as fct
import itertools as itt
import os

import cv2
import dask as da
import dask.array as darr
import networkx as nx
import numpy as np
import pandas as pd
import sparse
import xarray as xr
from scipy.ndimage.filters import median_filter
from scipy.ndimage.measurements import label
from scipy.signal import butter, lfilter
from scipy.stats import kstest, zscore
from skimage.morphology import disk
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree, radius_neighbors_graph

from .cnmf import adj_corr, graph_optimize_corr, label_connected
from .utilities import custom_arr_optimize, local_extreme, save_minian


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
        fm_max = local_extreme(fm, selem, diff=diff)
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
    # vectorized indexing on dask arrays produce a single chunk.
    # to memory issue, split seeds into 128 chunks, with chunk size no greater than 100
    chk_size = min(int(len(seeds) / 128), 100)
    vsub_ls = []
    for _, seed_sub in seeds.groupby(np.arange(len(seeds)) // chk_size):
        vsub = varr.sel(
            height=seed_sub["height"].to_xarray(), width=seed_sub["width"].to_xarray()
        )
        vsub_ls.append(vsub)
    varr_sub = xr.concat(vsub_ls, "index")
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
    # vectorized indexing on dask arrays produce a single chunk.
    # to memory issue, split seeds into 128 chunks, with chunk size no greater than 100
    chk_size = min(int(len(seeds) / 128), 100)
    vsub_ls = []
    for _, seed_sub in seeds.groupby(np.arange(len(seeds)) // chk_size):
        vsub = varr.sel(
            height=seed_sub["height"].to_xarray(), width=seed_sub["width"].to_xarray()
        )
        vsub_ls.append(vsub)
    varr_sub = xr.concat(vsub_ls, "index")
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
    print("computing distance")
    nng = radius_neighbors_graph(seeds[["height", "width"]], thres_dist)
    print("computing correlations")
    adj = adj_corr(varr, nng, seeds[["height", "width"]], noise_freq)
    print("merging seeds")
    adj = adj > thres_corr
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
    print("optimizing computation graph")
    nod_df = pd.DataFrame(
        np.array(
            list(
                itt.product(
                    np.arange(varr.sizes["height"]), np.arange(varr.sizes["width"])
                )
            )
        ),
        columns=["height", "width"],
    ).merge(seeds.reset_index(), how="outer", on=["height", "width"])
    seed_df = nod_df[nod_df["index"].notnull()]
    nn_tree = KDTree(nod_df[["height", "width"]], leaf_size=(2 * wnd) ** 2)
    nns_arr = nn_tree.query_radius(seed_df[["height", "width"]], r=wnd)
    sdg = nx.Graph()
    sdg.add_nodes_from(
        [
            (i, d)
            for i, d in enumerate(
                nod_df[["height", "width", "index"]].to_dict("records")
            )
        ]
    )
    for isd, nns in enumerate(nns_arr):
        cur_sd = seed_df.index[isd]
        sdg.add_edges_from([(cur_sd, n) for n in nns if n != cur_sd])
    sdg.remove_nodes_from(list(nx.isolates(sdg)))
    sdg = nx.convert_node_labels_to_integers(sdg)
    corr_df = graph_optimize_corr(varr, sdg, noise_freq)
    print("building spatial matrix")
    corr_df = corr_df[corr_df["corr"] > thres_corr]
    nod_df = pd.DataFrame.from_dict(dict(sdg.nodes(data=True)), orient="index")
    seed_df = nod_df[nod_df["index"].notnull()].astype({"index": int})
    A_ls = []
    Ashape = (varr.sizes["height"], varr.sizes["width"])
    for seed_id, sd in seed_df.iterrows():
        src_corr = corr_df[corr_df["target"] == seed_id].copy()
        src_nods = nod_df.loc[src_corr["source"]]
        src_corr["height"], src_corr["width"] = (
            src_nods["height"].values,
            src_nods["width"].values,
        )
        tgt_corr = corr_df[corr_df["source"] == seed_id].copy()
        tgt_nods = nod_df.loc[tgt_corr["target"]]
        tgt_corr["height"], tgt_corr["width"] = (
            tgt_nods["height"].values,
            tgt_nods["width"].values,
        )
        cur_corr = pd.concat([src_corr, tgt_corr]).append(
            {"corr": 1, "height": sd["height"], "width": sd["width"]}, ignore_index=True
        )
        cur_A = darr.array(
            sparse.COO(cur_corr[["height", "width"]].T, cur_corr["corr"], shape=Ashape)
        )
        A_ls.append(cur_A)
    A = xr.DataArray(
        darr.stack(A_ls).map_blocks(lambda a: a.todense(), dtype=float),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": seed_df["index"].values,
            "height": varr.coords["height"].values,
            "width": varr.coords["width"].values,
        },
    )
    return A


def initC(varr, A):
    uids = A.coords["unit_id"]
    fms = varr.coords["frame"]
    A = A.data.map_blocks(sparse.COO).map_blocks(lambda a: a / a.sum()).compute()
    C = darr.tensordot(A, varr, axes=[(1, 2), (1, 2)])
    C = xr.DataArray(
        C, dims=["unit_id", "frame"], coords={"unit_id": uids, "frame": fms}
    )
    return C


def initbf(varr, A, C):
    A = A.data.map_blocks(sparse.COO).compute()
    Yb = (varr - darr.tensordot(C, A, axes=[(0,), (0,)])).clip(0)
    b = Yb.mean("frame")
    f = Yb.mean(["height", "width"])
    arr_opt = fct.partial(
        custom_arr_optimize, rename_dict={"tensordot": "tensordot_restricted"}
    )
    with da.config.set(array_optimize=arr_opt):
        b = da.optimize(b)[0]
        f = da.optimize(f)[0]
    b, f = da.compute([b, f])[0]
    return b, f


@darr.as_gufunc(signature="(h, w)->(h, w)", output_dtypes=int, allow_rechunk=True)
def da_label(im):
    return label(im)[0]
