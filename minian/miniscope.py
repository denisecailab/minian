from __future__ import print_function
import numpy as np
import caiman as cm
import pylab as pl
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import itertools as itt
import scipy.misc as misc
import os
import glob
import re
import warnings

# import sparse
import time
import re
import functools
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import signal as sig
from scipy.ndimage.measurements import center_of_mass
from scipy.stats import sem
from caiman.source_extraction.cnmf import cnmf
from caiman import motion_correction, components_evaluation
from skvideo import io as sio

# from matplotlib_venn import venn2
from collections import deque, OrderedDict
from decimal import Decimal

# import numericbtree as nbtree


def align_across_session(a1, a2, fn_mov_rig1, fn_mov_rig2):
    """[summary]

    Args:
        a1 ([type]): [description]
        a2 ([type]): [description]
        fn_mov_rig1 ([type]): [description]
        fn_mov_rig2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    mov_rig1 = np.load(fn_mov_rig1, mmap_mode="r")
    mov_rig2 = np.load(fn_mov_rig2, mmap_mode="r")
    mov_rig1_mean = np.mean(mov_rig1, axis=0)
    mov_rig2_mean = np.mean(mov_rig2, axis=0)
    cross_corr = sig.fftconvolve(mov_rig1_mean, mov_rig2_mean, mode="same")
    maximum = np.unravel_index(np.argmax(cross_corr), mov_rig1_mean.shape)
    midpoints = mov_rig1_mean.shape / 2
    shifts = maximum - midpoints
    a2.reshape((mov_rig1_mean.shape, -1))
    a2 = np.roll(a2, shifts, (0, 1))
    return a1, a2, shifts


def estimate_overlap(
    a1, a2, dims=None, dist_cutoff=5, method="max", search_range=5, restrict_search=True
):
    """[summary]

    Args:
        a1 ([type]): [description]
        a2 ([type]): [description]
        dims ([type], optional): [description]. Defaults to None.
        dist_cutoff (int, optional): [description]. Defaults to 5.
        method (str, optional): [description]. Defaults to 'max'.
        search_range (int, optional): [description]. Defaults to 5.
        restrict_search (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if np.ndim(a1) < 3:
        a1 = a1.reshape(np.append(dims, [-1]), order="F")
        a2 = a2.reshape(np.append(dims, [-1]), order="F")
    centroids_a1 = np.zeros((a1.shape[2], 2))
    centroids_a2 = np.zeros((a2.shape[2], 2))
    dist_centroids = np.zeros((a1.shape[2], a2.shape[2]))
    for idca1, ca1 in enumerate(centroids_a1):
        centroids_a1[idca1, :] = center_of_mass(a1[:, :, idca1])
    for idca2, ca2 in enumerate(centroids_a2):
        centroids_a2[idca2, :] = center_of_mass(a2[:, :, idca2])
    for idxs, dist in np.ndenumerate(dist_centroids):
        print("calculating distance for pair: " + str(idxs))
        ca1 = centroids_a1[idxs[0]]
        ca2 = centroids_a2[idxs[1]]
        dist_centroids[idxs] = np.sqrt((ca1[0] - ca2[0]) ** 2 + (ca1[1] - ca2[1]) ** 2)
    dist_min0 = np.tile(
        np.min(dist_centroids, axis=1), (dist_centroids.shape[1], 1)
    ).transpose()
    dist_min0 = dist_centroids == dist_min0
    dist_min1 = np.tile(np.min(dist_centroids, axis=0), (dist_centroids.shape[0], 1))
    dist_min1 = dist_centroids == dist_min1
    dist_mask = np.logical_and(dist_min0, dist_min1)
    dist_cut = dist_centroids < dist_cutoff
    dist_mask = np.logical_and(dist_mask, dist_cut)
    correlations = np.zeros((a1.shape[2], a2.shape[2], 3))
    if method:
        min_idxs = list()
        if restrict_search:
            min_dist_a1 = np.argmin(dist_centroids, axis=1)
            min_dist_a2 = np.argmin(dist_centroids, axis=0)
            for ida1, ida2 in enumerate(min_dist_a1):
                min_idxs.append((ida1, ida2))
            for ida2, ida1 in enumerate(min_dist_a2):
                min_idxs.append((ida1, ida2))
        for idxs, corr in np.ndenumerate(correlations[:, :, 0]):
            if min_idxs and idxs not in min_idxs:
                correlations[idxs + (0,)] = -1
                continue
            else:
                print("calculating correlation for pair: " + str(idxs))
                if dist_centroids[idxs] < dist_cutoff:
                    if method == "max":
                        search_dims = (search_range * 2 + 1, search_range * 2 + 1)
                        corr_temp = np.zeros(search_dims)
                        for id_shift, corr_shift in np.ndenumerate(corr_temp):
                            shift = tuple(ish - search_range for ish in id_shift)
                            a1_temp = np.roll(a1[:, :, idxs[0]], shift[0], axis=0)
                            a1_temp = np.roll(a1_temp, shift[1], axis=1).flatten()
                            a2_temp = a2[:, :, idxs[1]].flatten()
                            corr_temp[id_shift] = np.corrcoef(a1_temp, a2_temp)[0, 1]
                        max_shift = np.unravel_index(np.argmax(corr_temp), search_dims)
                        correlations[idxs + (0,)] = np.max(corr_temp)
                        correlations[idxs + (1,)] = max_shift[0]
                        correlations[idxs + (2,)] = max_shift[1]
                    elif method == "plain":
                        a1_temp = a1[:, :, idxs[0]].flatten()
                        a2_temp = a2[:, :, idxs[1]].flatten()
                        correlations[idxs + (0,)] = np.corrcoef(a1_temp, a2_temp)[0, 1]
                    else:
                        print("Unrecognized method!")
                else:
                    correlations[idxs + (0,)] = -1
    nua1 = a1.shape[2]
    nua2 = a2.shape[2]
    ovlp = np.sum(dist_mask)
    return dist_centroids, dist_mask, correlations, (nua1, nua2, ovlp)


def infer_map_old(*args):
    """[summary]

    Returns:
        [type]: [description]
    """
    ovlp = np.eye(args[0].shape[0], dtype=bool)
    res = list()
    for idmask, mask in enumerate(args):
        if ovlp.shape[1] != mask.shape[0]:
            mask = mask.T
            if ovlp.shape[1] == mask.shape[0]:
                warnings.warn(
                    "dimension mismatch, using transpose of matrix " + str(idmask)
                )
            else:
                warnings.warn("dimension mismatch, skipping matrix " + str(idmask))
                continue
        new_ovlp = np.zeros((ovlp.shape[0], mask.shape[1]), dtype=bool)
        pairs_ovlp = np.nonzero(ovlp)
        pairs_mask = np.nonzero(mask)
        for idpovlp, povlp in enumerate(pairs_ovlp[1]):
            if povlp in pairs_mask[0]:
                idpmask = pairs_mask[1][np.where(pairs_mask[0] == povlp)]
                new_ovlp[pairs_ovlp[0][idpovlp], idpmask] = True
        ovlp = new_ovlp
    return ovlp


def calculate_centroids_old(*args):
    """[summary]

    Raises:
        AssertionError: [description]

    Returns:
        [type]: [description]
    """
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    centroids = list()
    for ida, cur_a in enumerate(args):
        print("calculating centroids for matrix " + str(ida))
        cur_centroid = np.zeros((nunits[ida], 2))
        for idu, u in enumerate(cur_centroid):
            cur_centroid[idu, :] = center_of_mass(cur_a[:, :, idu])
        centroids.append(cur_centroid)
    return centroids


def calculate_centroids_distance_old(*args, **kwargs):
    """[summary]

    Raises:
        AssertionError: [description]

    Returns:
        [type]: [description]
    """
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    dims = set([a.shape[0:2] for a in args])
    if len(dims) > 1:
        warnings.warn(
            "inputs dimensions mismatch. using the dimensions of first spatial matrix"
        )
    dims = dims.pop()
    centroids = kwargs.get("cent_in", list())
    if not centroids:
        centroids = calculate_centroids_old(*args)
    tile = kwargs.get("tile", None)
    if tile:
        cent0 = np.linspace(0, dims[0], np.ceil(dims[0] * 2.0 / tile[0]))
        cent1 = np.linspace(0, dims[1], np.ceil(dims[1] * 2.0 / tile[1]))
        coords = np.empty(shape=(len(nunits), 0))
        for cent in itt.product(cent0, cent1):
            print("center: " + str(cent))
            centroids_inrange = list()
            for cur_centroids in centroids:
                cur_inrange = cur_centroids[:, 0] > (cent[0] - np.ceil(tile[0] / 2.0))
                cur_inrange = np.logical_and(
                    cur_inrange,
                    cur_centroids[:, 0] < (cent[0] + np.ceil(tile[0] / 2.0)),
                )
                cur_inrange = np.logical_and(
                    cur_inrange,
                    cur_centroids[:, 1] > (cent[1] - np.ceil(tile[1] / 2.0)),
                )
                cur_inrange = np.logical_and(
                    cur_inrange,
                    cur_centroids[:, 1] < (cent[1] + np.ceil(tile[1] / 2.0)),
                )
                centroids_inrange.append(np.nonzero(cur_inrange)[0])
            cur_coords = np.empty(shape=(len(nunits), 0))
            for pair_inrange in itt.product(*centroids_inrange):
                pair_inrange = np.array(pair_inrange).reshape((len(nunits), -1))
                cur_coords = np.append(cur_coords, pair_inrange, axis=1)
            coords = np.hstack((coords, cur_coords))
        dist_centroids = sparse.COO(
            coords, data=np.array((-1,) * coords.shape[1]), shape=nunits
        )
    else:
        dist_centroids = np.zeros(nunits, dtype=np.float32)
    dist_it = np.nditer(dist_centroids, flags=["multi_index"], op_flags=["readwrite"])
    print("calculating centroids distance with shape: " + str(dist_centroids.shape))
    while not dist_it.finished:
        idx = dist_it.multi_index
        cur_centroids = np.array(
            [centroids[ida][idu, :] for ida, idu in enumerate(idx)]
        )
        midpoint = np.tile(np.mean(cur_centroids, axis=0), (len(cur_centroids), 1))
        dist_it[0] = np.sum(np.sqrt(np.sum((cur_centroids - midpoint) ** 2, axis=1)))
        dist_it.iternext()
    return dist_centroids


def estimate_threshold(*args):
    """[summary]

    Raises:
        AssertionError: [description]

    Returns:
        [type]: [description]
    """
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    if len(nunits) < 3:
        warnings.warn("less than 3 matrix provided. returning default threshold")
        return 5
    dist_list = []
    args_sh = deque(args)
    args_sh.rotate(-1)
    print("threshold estimation start")
    for s0, s1 in zip(args, args_sh):
        dist_list.append(calculate_centroids_distance(s0, s1))
    thres = 0
    while thres < 100:
        map_list = deque()
        for ids, (s0, s1) in enumerate(zip(args, args_sh)):
            cur_map = calculate_map_old(
                s0, s1, dist_in=dist_list[ids], threshold=thres + 1
            )
            map_list.append(cur_map)
        infer_list = []
        for _ in range(len(map_list)):
            cur_infer = infer_map(*map_list)
            infer_list.append(cur_infer)
            map_list.rotate(1)
        if any(not np.array_equal(*np.nonzero(im)) for im in infer_list):
            print("estimated threshold: " + str(thres))
            break
        else:
            thres += 1
    return thres


def calculate_map_old(*args, **kwargs):
    """[summary]

    Raises:
        AssertionError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    ndims = np.array(np.ndim(a) for a in args)
    if np.any(ndims < 3):
        raise AssertionError("not a spatial matrix. reshape first!")
    nunits = tuple(a.shape[-1] for a in args)
    method = kwargs.get("method", "perunit")
    thres = kwargs.get("threshold", None)
    centroids = kwargs.get("cent_in", list())
    dist_centroids = kwargs.get("dist_in", np.zeros(nunits))
    dist_map = np.ones(nunits, dtype=bool)
    if not dist_centroids.any():
        if not centroids:
            dist_centroids = calculate_centroids_distance_old(*args)
        else:
            dist_centroids = calculate_centroids_distance_old(*args, cent_in=centroids)
    if not thres:
        thres = estimate_threshold(*args)
    thres = len(nunits) * np.sqrt(
        thres ** 2 / (2 - 2 * np.cos(2 * np.pi / len(nunits)))
    )
    print("using threshold: " + str(thres))
    for axis in range(dist_centroids.ndim):
        cur_dist = dist_centroids.swapaxes(0, axis)
        cur_map = np.zeros_like(dist_map).swapaxes(0, axis)
        if method == "perunit":
            for uid, unit in enumerate(cur_dist):
                pid = np.unravel_index(np.argmin(unit), cur_map.shape[1:])
                cur_map[(uid,) + pid] = True
        elif method == "perpair":
            cur_min = np.argmin(cur_dist, axis=0)
            min_it = np.nditer(cur_min, flags=["multi_index"])
            while not min_it.finished:
                cur_map[(min_it[0],) + min_it.multi_index] = True
                min_it.iternext()
        else:
            raise ValueError("unrecognized method")
        cur_thres = cur_dist < thres
        cur_map = np.logical_and(cur_map, cur_thres)
        cur_map = cur_map.swapaxes(0, axis)
        dist_map = np.logical_and(dist_map, cur_map)
    return dist_map


def resolve_conflicts(meta_all, snames):
    """[summary]

    Args:
        meta_all ([type]): [description]
        snames ([type]): [description]

    Returns:
        [type]: [description]
    """
    fcutoff = 50
    # meta_all['nconflict'] = meta_all['conflict_with'].apply(lambda l: np.sum(l))
    # meta_all = meta_all.sort_values('nconflict', ascending=False)
    conf_pairs = meta_all[meta_all["conflict_with"].apply(bool)]["conflict_with"]
    conf_list = [[list(conf_pairs.index).index(c) for c in p] for p in conf_pairs]
    nconf = len(conf_list)
    conf_tree = nbtree.NBTree(nconf)
    remv_exp_last = 0
    print("processing conflict list with length {}".format(nconf))
    for level in range(fcutoff, nconf):
        t = time.time()
        comb_exp = np.sum(
            [misc.comb(level, posslen) for posslen in range(fcutoff, level + 1)]
        )
        remv_exp = comb_exp - remv_exp_last * 2
        remv_exp_last = comb_exp
        nremoved = 0
        nsearched = 0
        print("processing level: " + str(level))
        print("expected removal: " + str(remv_exp))
        for nd in conf_tree.get_nodes(level=level, unpack=True):
            print("searched: {0}, removed: {1}".format(nsearched, nremoved), end="\r")
            if nremoved >= remv_exp:
                break
            if nd % 2:
                nsearched += 1
                fal = np.array(conf_tree.path_to_node(nd)) % 2
                if np.sum(fal) >= fcutoff:
                    conf_tree.remove_subtree(nd)
                    nremoved += 1
        print("process time: {} s".format(time.time() - t))
    for conid1, pair in enumerate(conf_list):
        for conid2 in pair:
            conidl = min(conid1, conid2)
            conidh = max(conid1, conid2)
            print("processing pair:" + str((conidl, conidh)))
            for ndl in conf_tree.get_nodes(level=conidl + 1, unpack=True):
                if not ndl / 2:
                    for ndh in conf_tree.children(ndl).get_nodes(
                        level=conidh + 1, unpack=True
                    ):
                        if not ndh / 2:
                            conf_tree.remove_subtree(ndh)
    return conf_tree


def cut_nodes(meta_all, cutoff):
    """[summary]

    Args:
        meta_all ([type]): [description]
        cutoff ([type]): [description]

    Returns:
        [type]: [description]
    """
    conf_pairs = meta_all[meta_all["conflict_with"].apply(bool)]["conflict_with"]
    conf_list = [[list(conf_pairs.index).index(c) for c in p] for p in conf_pairs]
    conf_tree = nbtree.NBTree(len(conf_list))
    beg_node = conf_tree.leaves(0).begin()
    end_node = conf_tree.leaves(0).end()
    cur_node = beg_node
    processed = 0
    removed = 0
    while cur_node < end_node:
        if processed % 1000 == 0:
            print(
                "processing {0:.15E} th node. processed: {1}, removed: {2}, left: {3:.15E}".format(
                    cur_node - beg_node,
                    processed,
                    removed,
                    Decimal(end_node - cur_node),
                )
            )
        cur_node = conf_tree.get_nodes(level=conf_tree._depth, unpack=True).next()
        path = conf_tree.path_to_node(cur_node)
        pathodd = [p % 2 for p in path]
        iodd = -1
        try:
            for _ in range(cutoff):
                iodd = pathodd.index(1, iodd + 1)
        except ValueError:
            pass
        if iodd > 0:
            conf_tree.remove_subtree(path[iodd])
            removed += 1
        processed += 1
    return conf_tree


def plot_venn(sets, setlabels, savepath=""):
    """[summary]

    Args:
        sets ([type]): [description]
        setlabels ([type]): [description]
        savepath (str, optional): [description]. Defaults to ''.
    """
    pl.rcParams.update({"font.size": "19"})
    fig = pl.figure()
    nsets = len(sets)
    for setid, cur_set in enumerate(sets):
        ax = fig.add_subplot(nsets, 1, setid + 1)
        v = venn2(cur_set, set_labels=setlabels[setid], ax=ax)
        ratio0 = cur_set[2] / float(cur_set[0])
        ratio1 = cur_set[2] / float(cur_set[1])
        ratiomean = cur_set[2] / np.mean([cur_set[0], cur_set[1]])
        ratiosum = cur_set[2] / float(cur_set[0] + cur_set[1] - cur_set[2])
        a = setlabels[setid][0]
        b = setlabels[setid][1]
        pl.text(
            1,
            0.8,
            r"$\frac{"
            + a
            + r" \cap "
            + b
            + r"}{"
            + a
            + r"} = "
            + "{:.3}".format(ratio0)
            + r"$",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        pl.text(
            1,
            0.6,
            r"$\frac{"
            + a
            + r" \cap "
            + b
            + r"}{"
            + b
            + r"} = "
            + "{:.3}".format(ratio1)
            + r"$",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        pl.text(
            1,
            0.4,
            r"$\frac{"
            + a
            + r" \cap "
            + b
            + r"}{mean("
            + a
            + r", "
            + b
            + r")} = "
            + "{:.3}".format(ratiomean)
            + r"$",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        pl.text(
            1,
            0.2,
            r"$\frac{"
            + a
            + r" \cap "
            + b
            + r"}{"
            + a
            + r" \cup "
            + b
            + r"} = "
            + "{:.3}".format(ratiosum)
            + r"$",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_title(a + " vs " + b)

        # if __name__ == '__main__':
        # a1 = np.load('/media/share/Denise/Wired Valence/Wired Valence Organized Data/MS101/4/H11_M52_S45/cnm.npz')['A']
        # a2 = np.load('/media/share/Denise/Wired Valence/Wired Valence Organized Data/MS101/5/H11_M41_S56/cnm.npz')['A']
        # dims = np.load('/media/share/Denise/Wired Valence/Wired Valence Organized Data/MS101/4/H11_M52_S45/cnm.npz')['dims']
