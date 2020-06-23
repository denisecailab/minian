import functools
import itertools as itt
import os
import re
import time
import warnings
from collections import OrderedDict

import numpy as np
import os.path
import pandas as pd
import xarray as xr
from scipy.ndimage.measurements import center_of_mass


def batch_load_spatial_component(animalpath):
    """[summary]

    Args:
        animalpath ([type]): [description]

    Returns:
        [type]: [description]
    """
    sa = OrderedDict()
    for dirname, subdirs, files in os.walk(animalpath):
        if files:
            if os.path.isfile(dirname + os.sep + "cnm.npz"):
                dirnamelist = dirname.split(os.sep)
                cur_a = np.load(dirname + os.sep + "cnm.npz")["A"]
                dims = np.load(dirname + os.sep + "cnm.npz")["dims"]
                cur_a = np.reshape(cur_a, np.append(dims, (-1,)), order="F")
                sessionid = "s" + dirnamelist[-2]
                cur_a = xr.DataArray(
                    cur_a,
                    coords={
                        "ay": range(cur_a.shape[0]),
                        "ax": range(cur_a.shape[1]),
                        "unitid": range(cur_a.shape[2]),
                    },
                    dims=("ay", "ax", "unitid"),
                    name=sessionid,
                )
                sa[sessionid] = cur_a
                print("loading: " + dirname)
            else:
                print("cnm not found in folder: " + dirname + " proceed")
        else:
            print("empty folder: " + dirname + " proceed")
    return sa


def batch_calculate_centroids(sa):
    """[summary]

    Args:
        sa ([type]): [description]

    Returns:
        [type]: [description]
    """
    cent_dict = OrderedDict()
    for sid, a in sa.items():
        cent_dict[sid] = calculate_centroids(a)
    return cent_dict


def calculate_centroids(a):
    """[summary]

    Args:
        a ([type]): [description]

    Returns:
        [type]: [description]
    """
    print("calculating centroids for " + a.name)
    centroids = np.zeros((a.shape[2], 2))
    for idu, u in enumerate(centroids):
        centroids[idu, :] = center_of_mass(a.values[:, :, idu])
    centroids = xr.DataArray(
        centroids.T,
        coords={"centloc": ["cy", "cx"], "unitid": range(a.shape[2])},
        dims=("centloc", "unitid"),
        name=a.name,
    )
    return centroids


def batch_calculate_centroids_distances(cent, dims, tile=None):
    """[summary]

    Args:
        cent ([type]): [description]
        dims ([type]): [description]
        tile ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    t_start = time.time()
    dist_dict = OrderedDict()
    for comblen in range(2, len(cent) + 1):
        for comb in itt.combinations(cent.items(), comblen):
            sidset = tuple([itpair[0] for itpair in comb])
            centset = dict(comb)
            print("calculating distance of centroids for sessions:" + str(sidset))
            dist = calculate_centroids_distance(centset, dims, tile)
            dist_dict[sidset] = dist
    print("total running time:" + str(time.time() - t_start))
    return dist_dict


def calculate_centroids_distance(centroids, dims, tile=None):
    """[summary]

    Args:
        centroids ([type]): [description]
        dims ([type]): [description]
        tile ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    nunits = tuple(a.shape[-1] for a in centroids.values())
    print("number of units: " + str(nunits))
    if not tile:
        tile = dims
    centy = np.linspace(0, dims[0], np.ceil(dims[0] * 2.0 / tile[0]) + 1)
    centx = np.linspace(0, dims[1], np.ceil(dims[1] * 2.0 / tile[1]) + 1)
    dy = np.ceil(tile[0] / 2.0)
    dx = np.ceil(tile[1] / 2.0)
    coords = list()
    centroids_y = dict(
        [(sid, sa.sel(centloc="cy").values) for (sid, sa) in centroids.items()]
    )
    centroids_x = dict(
        [(sid, sa.sel(centloc="cx").values) for (sid, sa) in centroids.items()]
    )
    for cy, cx in itt.product(centy, centx):
        centroids_inrange = list()
        for sid, cur_cent in centroids.items():
            inrangey = np.logical_and(
                centroids_y[sid] > (cy - dy), centroids_y[sid] < (cy + dy)
            )
            inrangex = np.logical_and(
                centroids_x[sid] > (cx - dx), centroids_x[sid] < (cx + dx)
            )
            inrange = np.logical_and(inrangex, inrangey)
            centroids_inrange.append(np.nonzero(inrange)[0].tolist())
        for pair in itt.product(*centroids_inrange):
            coords.append(pair)
    mulidx = pd.MultiIndex.from_tuples(coords, names=centroids.keys())
    mulidx = mulidx.drop_duplicates()
    print("subsetting " + str(mulidx.shape[0]) + " pairs")
    dist = np.array([])
    for idx in mulidx:
        cur_centroids = np.array(
            [centroids[centroids.keys()[ids]][:, idu] for ids, idu in enumerate(idx)]
        )
        midpoint = np.tile(np.mean(cur_centroids, axis=0), (len(cur_centroids), 1))
        dist = np.append(
            dist, np.sum(np.sqrt(np.sum((cur_centroids - midpoint) ** 2, axis=1)))
        )
    return pd.Series(dist, mulidx, name="distance")


def batch_calculate_map(dist):
    """[summary]

    Args:
        dist ([type]): [description]

    Returns:
        [type]: [description]
    """
    map_dict = OrderedDict()
    for sname, d in dist.items():
        map_dict[sname] = calculate_map(d)
    return map_dict


def calculate_map(dist):
    """[summary]

    Args:
        dist ([type]): [description]

    Returns:
        [type]: [description]
    """
    minidx = dist.index
    for sname in dist.index.names:
        cur_minidx = dist.groupby(level=sname).apply(lambda d: d.argmin())
        minidx = minidx.intersection(cur_minidx)
    dist_map = dist[minidx]
    return dist_map


def initialize_meta_map(map_dict):
    """[summary]

    Args:
        map_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    snames = list(map_dict.keys()[np.argmax(map(len, map_dict.keys()))])
    meta_all = pd.DataFrame(
        pd.concat([m.reset_index() for m in map_dict.values()], ignore_index=True)
    )
    meta_all = reset_meta_map(meta_all, snames)
    meta_all = meta_all.sort_values("nsession", ascending=False)
    update_meta_map(meta_all, snames)
    return meta_all, snames


def reset_meta_map(meta_all, snames=None):
    """[summary]

    Args:
        meta_all ([type]): [description]
        snames ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if not snames:
        snames = meta_all.sessions
    meta_all[snames] = meta_all[snames].apply(pd.to_numeric, downcast="integer")
    meta_all["nsession"] = meta_all[snames].count(axis=1)
    meta_all["missing"] = [[]] * len(meta_all)
    meta_all["conflict"] = [[]] * len(meta_all)
    meta_all["conflict_with"] = [[]] * len(meta_all)
    meta_all["match_score"] = np.nan
    meta_all["missing_score"] = np.nan
    meta_all["conflict_score"] = np.nan
    meta_all["score"] = np.nan
    meta_all["active"] = True
    return meta_all


def update_meta_map(meta_all, snames):
    """[summary]

    Args:
        meta_all ([type]): [description]
        snames ([type]): [description]
    """
    row_count = 0
    for cur_idx, cur_entry in meta_all.iterrows():
        row_count += 1
        if row_count % 100 == 0:
            print("iteration: {} out of {}".format(row_count, meta_all.shape[0]))
        cur_sname = cur_entry[snames].dropna().index
        cur_block = meta_all[meta_all["active"]]
        cur_block = cur_block[
            pd.DataFrame(meta_all == cur_entry)[cur_sname].any(axis=1)
        ]
        sub_block = cur_block[cur_block["nsession"] <= cur_entry["nsession"]]
        sub_block_nm = ~sub_block.apply(
            lambda r: (r == cur_entry).replace(~r.isnull(), True), axis=1
        )[cur_sname]
        cur_nm = sub_block[sub_block_nm.apply(lambda r: r.any(), axis=1)]
        cur_conf = ~cur_block.apply(
            lambda r: (r == cur_entry).replace(~r.isnull(), True), axis=1
        )[cur_sname]
        cur_conf = cur_block[cur_conf.apply(lambda r: r.any(), axis=1)]
        cur_conf = set(cur_conf.index) - {cur_idx}
        cur_exp = set(
            frozenset(cmb)
            for slen in range(2, len(cur_sname))
            for cmb in itt.combinations(cur_sname, slen)
        )
        cur_block_sname = set(
            sub_block.apply(lambda r: frozenset(r[cur_sname].dropna().index), axis=1)
        )
        cur_ms = cur_exp - cur_block_sname
        meta_all.set_value(
            cur_idx,
            "missing",
            meta_all.loc[cur_idx, "missing"] + [tuple(m) for m in cur_ms],
        )
        meta_all.set_value(
            cur_idx, "conflict", meta_all.loc[cur_idx, "conflict"] + list(cur_nm.index)
        )
        meta_all.loc[cur_conf, "conflict_with"] = meta_all.loc[
            cur_conf, "conflict_with"
        ].apply(lambda r: r + [cur_idx])
        match_score = sub_block.apply(lambda r: r == cur_entry, axis=1)[
            cur_sname
        ].values.sum()
        missing_score = len(
            list(itt.chain.from_iterable(meta_all.loc[cur_idx, "missing"]))
        )
        conflict_score = np.sum(sub_block_nm.values)
        score = match_score - conflict_score
        meta_all.set_value(cur_idx, "match_score", match_score)
        meta_all.set_value(cur_idx, "missing_score", missing_score)
        meta_all.set_value(cur_idx, "conflict_score", conflict_score)
        meta_all.set_value(cur_idx, "score", score)


def threshold_meta_map(meta_all, threshold):
    """[summary]

    Args:
        meta_all ([type]): [description]
        threshold ([type]): [description]

    Returns:
        [type]: [description]
    """
    thres = meta_all["nsession"].apply(
        lambda x: x * np.sqrt(threshold ** 2 / (2 - 2 * np.cos(2 * np.pi / x)))
    )
    return meta_all[meta_all["distance"] < thres]


def group_meta_map(meta_all):
    """[summary]

    Args:
        meta_all ([type]): [description]

    Returns:
        [type]: [description]
    """
    meta_grp = meta_all.copy()
    grplen = meta_grp["group"].apply(lambda l: len(l))
    for cur_idx, cur_entry in meta_grp[grplen > 1].iterrows():
        cand = cur_entry["group"]
        cand_scr = np.array(
            [
                meta_grp.loc[can, "match_score"]
                if not meta_grp.loc[can, "conflict_score"]
                else 0
                for can in cand
            ]
        )
        meta_grp.set_value(
            cur_idx,
            "group",
            [cand[ic] for ic in np.where(cand_scr == cand_scr.min())[0]],
        )
    return meta_grp


def subset_data_by_list(data, column, tlist):
    """[summary]

    Args:
        data ([type]): [description]
        column ([type]): [description]
        tlist ([type]): [description]

    Returns:
        [type]: [description]
    """
    return data[data[column].apply(lambda l: bool(set(l) & set(tlist)))]


def subset_data_by_map(meta_all, snames):
    """[summary]

    Args:
        meta_all ([type]): [description]
        snames ([type]): [description]

    Returns:
        [type]: [description]
    """
    snames = list(snames)
    snames_all = filter(lambda x: re.match("s[0-9]+$", x), meta_all.columns)
    snames_nan = list(set(snames_all) - set(snames))
    meta_null = meta_all[snames_all].isnull()
    satisfied = meta_all[
        meta_null.apply(lambda r: ~r[snames].any() and r[snames_nan].all(), axis=1)
    ]
    satisfied.sessions = snames
    return satisfied


def infer_meta_map(meta_all, snames=None):
    """[summary]

    Args:
        meta_all ([type]): [description]
        snames ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if not snames:
        try:
            snames = meta_all.sessions
        except AttributeError:
            snames = filter(lambda x: re.match("s[0-9]+$", x), meta_all.columns)
    infer_list = []
    for nsession in range(3, len(snames) + 1):
        for sessions_infer in itt.combinations(snames, nsession):
            print("inferring " + str(sessions_infer))
            infer = infer_map(meta_all, sessions_infer)
            infer_list.append(infer)
    pair_list = [subset_data_by_map(meta_all, s) for s in itt.combinations(snames, 2)]
    meta_infer = pd.concat(infer_list + pair_list, ignore_index=True)
    meta_infer = meta_infer.reindex_axis(meta_all.columns, axis=1)
    meta_infer = reset_meta_map(meta_infer, snames)
    meta_infer.sessions = snames
    return meta_infer


def infer_map(meta_all, snames):
    """[summary]

    Args:
        meta_all ([type]): [description]
        snames ([type]): [description]

    Returns:
        [type]: [description]
    """
    meta_pair_list = [
        subset_data_by_map(meta_all, s) for s in itt.combinations(snames, 2)
    ]
    meta_infer = filter(lambda m: snames[0] in m.sessions, meta_pair_list)[0]
    for cur_on in snames:
        cur_maps = filter(lambda m: cur_on in m.sessions, meta_pair_list)
        for cur_m in cur_maps:
            meta_infer = extend_map(meta_infer, cur_m, cur_on)
    meta_infer.sessions = snames
    return meta_infer


def extend_map(mapx, mapy, on):
    """[summary]

    Args:
        mapx ([type]): [description]
        mapy ([type]): [description]
        on ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not hasattr(on, "__iter__"):
        on = [on]
    try:
        inter = mapx[on].reset_index().merge(mapy[on].reset_index(), on=on)
    except KeyError:
        inter = pd.DataFrame()
    extended = pd.DataFrame()
    ext_sessions = list(set(mapx.sessions).union(set(mapy.sessions)))
    for inter_idx, inter_row in inter.iterrows():
        sx = mapx.loc[inter_row.loc["index_x"], mapx.sessions]
        sy = mapy.loc[inter_row.loc["index_y"], mapy.sessions]
        extrow = pd.concat([sx, sy]).drop_duplicates()
        if len(extrow) <= len(ext_sessions):
            extended = extended.append(extrow, ignore_index=True)
    extended.sessions = ext_sessions
    return extended


def generate_summary(mapdict, sadict):
    """[summary]

    Args:
        mapdict ([type]): [description]
        sadict ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if len(mapdict) != len(sadict):
        raise ValueError("length of mappings and spatial components mismatch!")
    exp_meta = pd.concat(mapdict, names=["animal", "original_index"]).reset_index()
    snames = list(filter(lambda x: re.match("s[0-9]+$", x), exp_meta.columns))
    exp_meta["sessions"] = exp_meta[snames].apply(
        lambda r: tuple(r[r.notnull()].index.tolist()), axis=1
    )
    exp_meta = exp_meta.groupby("animal").apply(group_by_session)
    summary = (
        exp_meta.groupby(["animal", "grouping_by_session", "sessions"])
        .size()
        .reset_index(name="count")
    )
    grouping_dict = dict(
        summary.groupby(["animal", "grouping_by_session"]).groups.keys()
    )
    for cur_anm, cur_sa in sadict.items():
        for cur_session, cur_s in cur_sa.items():
            summary = summary.append(
                pd.Series(
                    {
                        "animal": cur_anm,
                        "grouping_by_session": grouping_dict[cur_anm],
                        "sessions": (cur_session,),
                        "count": len(cur_s.unitid),
                    }
                ),
                ignore_index=True,
            )
    return summary, exp_meta


def generate_overlap(summary, denominator="mean"):
    """[summary]

    Args:
        summary ([type]): [description]
        denominator (str, optional): [description]. Defaults to 'mean'.

    Returns:
        [type]: [description]
    """
    summary_map = summary[summary["sessions"].apply(lambda x: len(x)) > 1]
    calculation = functools.partial(calculate_overlap, summary=summary, on=denominator)
    overlaps = (
        summary_map.groupby(["animal", "sessions"])
        .apply(calculation)
        .reset_index(drop=True)
    )
    return overlaps


def calculate_overlap(mapping, summary, on="mean"):
    """[summary]

    Args:
        mapping ([type]): [description]
        summary ([type]): [description]
        on (str, optional): [description]. Defaults to 'mean'.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if isinstance(mapping, pd.DataFrame):
        if len(mapping) > 1:
            raise ValueError("can only handle one mapping at a time!")
        else:
            mapping = mapping.iloc[0]
    summary_single = summary[summary["sessions"].apply(lambda x: len(x)) == 1]
    summary_single = summary_single[summary_single["animal"] == mapping["animal"]]
    summary_single.loc[:, "sessions"] = summary_single["sessions"].apply(set)
    cur_snames = set(mapping["sessions"])
    cur_dev = summary_single[
        summary_single["sessions"].apply(lambda s: s <= cur_snames)
    ]
    cur_dev.loc[:, "sessions"] = cur_dev["sessions"].apply(tuple)
    if on == "mean":
        overlap = pd.Series()
        overlap["group"] = mapping["group"]
        overlap["animal"] = mapping["animal"]
        overlap["mappings"] = mapping["sessions"]
        overlap["on"] = "mean"
        overlap["freq"] = mapping["count"] * 1.0 / np.mean(cur_dev["count"])
    elif on == "each":
        overlap = pd.DataFrame()
        overlap["group"] = cur_dev["group"]
        overlap["animal"] = cur_dev["animal"]
        overlap["mappings"] = [mapping["sessions"]] * len(cur_dev)
        overlap["on"] = cur_dev["sessions"]
        overlap["freq"] = mapping["count"] * 1.0 / cur_dev["count"]
    else:
        warnings.warn("unrecognized denominator. Nothing calculated")
    return overlap


def group_by_session(group):
    """[summary]

    Args:
        group ([type]): [description]

    Returns:
        [type]: [description]
    """
    cur_snames = set()
    for ss in group.sessions.unique():
        cur_snames.update(ss)
    group["grouping_by_session"] = [tuple(sorted(cur_snames))] * len(group)
    return group


def construct_temporal_component(dpath, meta, include_single=True):
    """[summary]

    Args:
        dpath ([type]): [description]
        meta ([type]): [description]
        include_single (bool, optional): [description]. Defaults to True.

    Raises:
        FileNotFoundError: [description]
        FileNotFoundError: [description]
        FileNotFoundError: [description]

    Returns:
        [type]: [description]
    """
    snames = list(filter(lambda x: re.match("s[0-9]+$", x), meta.columns))
    temp_comp_meta = list()
    animal_list = list()
    for animal, group in meta.groupby("animal"):
        if os.path.exists(os.path.join(dpath, animal)):
            animalpath = os.path.join(dpath, animal)
        elif os.path.exists(os.path.join(dpath, animal.upper())):
            animalpath = os.path.join(dpath, animal.upper())
        else:
            raise FileNotFoundError(animal + " not found in data path")
        meta_session = group[snames].dropna(axis=1, how="all")
        temp_comp_list = OrderedDict()
        temp_com_len = []
        if include_single:
            single_list = list()
        for s in meta_session.columns:
            snumber = re.findall(r"\d+", s)[0]
            if os.path.exists(os.path.join(animalpath, snumber)):
                sessionpath = os.path.join(animalpath, snumber)
            else:
                raise FileNotFoundError(snumber + " not found in animal path")
            cnmfile = None
            for dirpath, dirnames, files in os.walk(sessionpath):
                if "cnm.npz" in files:
                    cnmfile = os.path.join(dirpath, "cnm.npz")
            if cnmfile:
                cur_c = np.load(cnmfile)["C"]
                temp_comp_list[s] = cur_c
                temp_com_len.append(cur_c.shape[1])
                if include_single:
                    cur_single = pd.Series(
                        list(set(np.arange(cur_c.shape[0] - 1)) - set(meta_session[s]))
                    )
                    single_list.append(pd.DataFrame({s: cur_single, "animal": animal}))
            else:
                raise FileNotFoundError("cnm file not found in " + sessionpath)
        if include_single:
            meta = pd.concat([meta] + single_list, ignore_index=True)
            meta_session = meta[meta.animal == animal][snames].dropna(axis=1, how="all")
        cur_t = np.empty(
            [len(meta_session.index), max(temp_com_len), len(meta_session.columns)]
        )
        cur_t[:] = np.nan
        cur_temp = xr.DataArray(
            cur_t,
            coords=[
                meta_session.index,
                np.arange(max(temp_com_len)),
                meta_session.columns.tolist(),
            ],
            dims=["mapping_id", "frame", "session_id"],
        )
        for sid, session in cur_temp.groupby("session_id"):
            cur_session = temp_comp_list[sid]
            for mid, temp in session.groupby("mapping_id"):
                uid = meta.iloc[mid][sid]
                if np.isnan(uid):
                    continue
                else:
                    uid = int(uid)
                t = cur_session[uid]
                cur_temp.loc[mid, 0 : len(t) - 1, sid] = t
        temp_comp_meta.append(cur_temp)
        animal_list.append(animal)
    temp_comp_aligned = xr.align(*temp_comp_meta, join="outer")
    return xr.concat(temp_comp_aligned, dim=pd.Index(animal_list, name="animal")), meta


def calculate_temporal_correlation(temp_comp):
    """[summary]

    Args:
        temp_comp ([type]): [description]

    Returns:
        [type]: [description]
    """
    temp_squeezed = temp_comp.dropna("mapping_id", how="all").dropna("frame", how="all")
    temp_pd = temp_squeezed.transpose("mapping_id", "frame").to_pandas()
    return temp_pd.corr()


def corr2_coeff(A, B):
    """[summary]

    Args:
        A ([type]): [description]
        B ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def corr2_coeff_xr(A, B, along, across, unique_dim=True, auto_reindex=True):
    """[summary]

    Args:
        A ([type]): [description]
        B ([type]): [description]
        along ([type]): [description]
        across ([type]): [description]
        unique_dim (bool, optional): [description]. Defaults to True.
        auto_reindex (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    A = A.dropna(along, how="all").dropna(across, how="all").transpose(across, along)
    B = B.dropna(along, how="all").dropna(across, how="all").transpose(across, along)
    if unique_dim:
        nameA = str(A.coords["session_id"].data)
        nameB = str(B.coords["session_id"].data)
        if nameA == nameB:
            nameA += "A"
            nameB += "B"
    else:
        nameA = "A"
        nameB = "B"
    A = A.rename({across: across + "_" + nameA})
    B = B.rename({across: across + "_" + nameB})
    if auto_reindex:
        A.coords[along] = np.arange(len(A.coords[along]))
        B.coords[along] = np.arange(len(B.coords[along]))
        if not np.array_equal(A.coords[along], B.coords[along]):
            warnings.warn(
                "Dimension along which coeff is calculated mismatch!"
                "re-examine the validility of data!"
            )
    A_mA = A - A.mean(along)
    B_mB = B - B.mean(along)
    ssA = np.square(A_mA).sum(along).expand_dims("T")
    ssB = np.square(B_mB).sum(along).expand_dims("T")
    return A_mA.dot(B_mB) / np.sqrt(ssA.dot(ssB))


def divide_sessions_old(temp_comp, div_dict):
    """[summary]

    Args:
        temp_comp ([type]): [description]
        div_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    temp_comp_div_list = []
    len_frame = len(temp_comp.coords["frame"])
    for s_orig, s_group in temp_comp.groupby("session_id"):
        if s_orig in div_dict.keys():
            div = div_dict[s_orig]
            s_new_array = np.empty(len_frame, dtype="U10")
            sd_new_array = np.empty(len_frame)
            nan_mask = np.full(len_frame, False)
            for s_new, sd_new in div.items():
                b, e = sd_new
                s_new_array[b:e] = s_new
                sd_new_array[b:e] = np.arange(e - b)
                nan_mask[b:e] = True
            s_new_array[nan_mask] = "trivial"
            sd_new_array[nan_mask] = np.arange(np.sum(nan_mask))
            idx_new = pd.MultiIndex.from_arrays(
                [s_new_array, sd_new_array], names=["segment_id", "frame_split"]
            )
            s_group.coords["frame"] = idx_new
        else:
            s_new_array = np.full(len_frame, "all")
            idx_new = pd.MultiIndex.from_arrays(
                [s_new_array, s_group.coords["frame"]],
                names=["segment_id", "frame_split"],
            )
            s_group.coords["frame"] = idx_new
        s_group.unstack("frame").rename({"frame_split": "frame"})
        print("finished unstacking for" + s_orig)
        temp_comp_div_list.append(s_group)
    temp_comp_div = xr.align(
        *temp_comp_div_list, copy=False, join="outer", exclude=("animal", "mapping_id")
    )
    return xr.concat(temp_comp_div, dim="session_id")


def divide_sessions(temp_comp, div_dict, sort_dict, grp_dict):
    """[summary]

    Args:
        temp_comp ([type]): [description]
        div_dict ([type]): [description]
        sort_dict ([type]): [description]
        grp_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    temp_comp = temp_comp.to_dataset(name="temp_comp")
    seg_array = np.full(
        (len(temp_comp.coords["session_id"]), len(temp_comp.coords["frame"])),
        "unclassified",
        dtype="U20",
    )
    seg_array = xr.DataArray(
        seg_array,
        dims=("session_id", "frame"),
        coords={
            "session_id": temp_comp.coords["session_id"],
            "frame": temp_comp.coords["frame"],
        },
    )
    segments = []
    for s_orig, div in div_dict.items():
        for seg, dur in div.items():
            b, e = dur
            seg_array.loc[{"session_id": s_orig, "frame": slice(b, e)}] = seg
            segments.append((s_orig, seg))
    temp_comp.coords["segment_id"] = (("session_id", "frame"), seg_array)
    d_ss = sort_dict.get("session_id", dict())
    d_sg = sort_dict.get("segment_id", dict())
    temp_comp.attrs["segments"] = sorted(
        segments, key=lambda t: (d_ss.get(t[0], t[0]), d_sg.get(t[1], t[1]))
    )
    temp_comp.attrs["div_dict"] = div_dict
    temp_comp.attrs["sort_dict"] = sort_dict
    temp_comp.attrs["grp_dict"] = grp_dict
    return temp_comp


def reduce_along_frame(temp_comp, quant):
    """[summary]

    Args:
        temp_comp ([type]): [description]
        quant ([type]): [description]

    Returns:
        [type]: [description]
    """
    segments = temp_comp["segment_id"]
    reduced = np.full(
        (
            len(temp_comp["animal"]),
            len(temp_comp["mapping_id"]),
            len(temp_comp["session_id"]),
            len(np.unique(segments.values)),
        ),
        np.nan,
    )
    reduced = xr.DataArray(
        reduced,
        dims=["animal", "mapping_id", "session_id", "segment_id"],
        coords={
            "animal": temp_comp["animal"],
            "mapping_id": temp_comp["mapping_id"],
            "session_id": temp_comp["session_id"],
            "segment_id": np.unique(segments.values),
        },
    )
    for cur_ss, grp_ss in segments.groupby("session_id"):
        for cur_seg in np.unique(grp_ss):
            cur_mask = grp_ss == cur_seg
            for cur_anm, grp_anm in reduced.groupby("animal"):
                cur_dat = (
                    temp_comp.data_vars["temp_comp"]
                    .loc[{"animal": cur_anm, "session_id": cur_ss, "frame": cur_mask}]
                    .dropna("frame", how="all")
                    .dropna("mapping_id", how="all")
                )
                if cur_dat.size > 0:
                    thres = cur_dat.quantile(quant)
                    reduced.loc[
                        {
                            "animal": cur_anm,
                            "mapping_id": cur_dat.coords["mapping_id"],
                            "session_id": cur_ss,
                            "segment_id": cur_seg,
                        }
                    ] = (cur_dat.max("frame") > thres)
    return reduced


def compute_correlations(temp_comp, along, across):
    """[summary]

    Args:
        temp_comp ([type]): [description]
        along ([type]): [description]
        across ([type]): [description]

    Returns:
        [type]: [description]
    """
    segments = temp_comp.attrs["segments"]
    corr_list = []
    for cur_anm in temp_comp["animal"].values:
        for comb in itt.combinations_with_replacement(segments, 2):
            dat_A = temp_comp.sel(animal=cur_anm, session_id=comb[0][0])
            dat_A = (
                dat_A.where(dat_A["segment_id"] == comb[0][1], drop=True)
                .to_array()
                .drop("segment_id")
                .squeeze("variable", drop=True)
            )
            dat_B = temp_comp.sel(animal=cur_anm, session_id=comb[1][0])
            dat_B = (
                dat_B.where(dat_B["segment_id"] == comb[1][1], drop=True)
                .to_array()
                .drop("segment_id")
                .squeeze("variable", drop=True)
            )
            if dat_A.size > 0 and dat_B.size > 0:
                print(
                    "computing correlation of {} with {} for animal {}".format(
                        comb[0], comb[1], cur_anm
                    )
                )
                cur_corr = corr2_coeff_xr(dat_A, dat_B, along, across)
                cur_corr.coords["session_id_A"] = comb[0][0]
                cur_corr.coords["session_id_B"] = comb[1][0]
                cur_corr.coords["segment_id_A"] = comb[0][1]
                cur_corr.coords["segment_id_B"] = comb[1][1]
    print("merging")
    return xr.merge(corr_list)


def count_temporal_overlap(reduced, grp_dict, sort_dict):
    """[summary]

    Args:
        reduced ([type]): [description]
        grp_dict ([type]): [description]
        sort_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    reduced = reduced.to_series().dropna()
    overlaps_list = []
    for cur_anm, grp_anm in reduced.groupby("animal"):
        segments = set(grp_anm.reset_index(["animal", "mapping_id"]).index.tolist())
        d_ss = sort_dict.get("session_id", dict())
        d_sg = sort_dict.get("segment_id", dict())
        segments = sorted(
            segments, key=lambda t: (d_ss.get(t[0], t[0]), d_sg.get(t[1], t[1]))
        )
        for comb in itt.combinations(segments, 2):
            dat_1 = grp_anm.loc[:, :, comb[0][0], comb[0][1]]
            dat_2 = grp_anm.loc[:, :, comb[1][0], comb[1][1]]
            inter = pd.concat([dat_1, dat_2], axis=1).fillna(0).all(axis=1)
            c_1 = dat_1.sum()
            c_2 = dat_2.sum()
            ints = inter.sum()
            overlaps_list.append(
                pd.Series(
                    {
                        "inter": comb,
                        "on": comb[0],
                        "group": grp_dict[cur_anm],
                        "animal": cur_anm,
                        "freq": ints * 1.0 / c_1,
                    }
                )
            )
            overlaps_list.append(
                pd.Series(
                    {
                        "inter": comb,
                        "on": comb[1],
                        "group": grp_dict[cur_anm],
                        "animal": cur_anm,
                        "freq": ints * 1.0 / c_2,
                    }
                )
            )
    overlaps = pd.concat(overlaps_list, axis=1).T.set_index(["inter", "on"])
    return overlaps


def count_active_cells(reduced):
    """[summary]

    Args:
        reduced ([type]): [description]

    Returns:
        [type]: [description]
    """
    active = reduced.sum("mapping_id").to_dataframe(name="active").reset_index()
    active["group"] = active["animal"].replace(grp_dict)
    return active
