import itertools as itt

import dask as da
import numpy as np
import pandas as pd

from .visualization import centroid


def calculate_centroids(A, window):
    A = A.where(window, 0)
    return centroid(A, verbose=True)


def calculate_centroid_distance(
    cents, by="session", index_dim=["animal"], tile=(50, 50)
):
    res_list = []

    def cent_pair(grp):
        dist_df_ls = []
        len_df = 0
        for (byA, grpA), (byB, grpB) in itt.combinations(list(grp.groupby(by)), 2):
            cur_pairs = subset_pairs(grpA, grpB, tile)
            pairs_ls = list(cur_pairs)
            len_df = len_df + len(pairs_ls)
            subA = grpA.set_index("unit_id").loc[[p[0] for p in pairs_ls]].reset_index()
            subB = grpB.set_index("unit_id").loc[[p[1] for p in pairs_ls]].reset_index()
            dist = da.delayed(pd_dist)(subA, subB).rename("distance")
            dist_df = da.delayed(pd.concat)(
                [subA["unit_id"].rename(byA), subB["unit_id"].rename(byB), dist],
                axis="columns",
            )
            dist_df = dist_df.rename(
                columns={
                    "distance": ("variable", "distance"),
                    byA: (by, byA),
                    byB: (by, byB),
                }
            )
            dist_df_ls.append(dist_df)
        dist_df = da.delayed(pd.concat)(dist_df_ls, ignore_index=True, sort=True)
        return dist_df, len_df

    print("creating parallel schedule")
    if index_dim:
        for idxs, grp in cents.groupby(index_dim):
            dist_df, len_df = cent_pair(grp)
            if type(idxs) is not tuple:
                idxs = (idxs,)
            meta_df = pd.concat(
                [
                    pd.Series([idx] * len_df, name=("meta", dim))
                    for idx, dim in zip(idxs, index_dim)
                ],
                axis="columns",
            )
            res_df = da.delayed(pd.concat)([meta_df, dist_df], axis="columns")
            res_list.append(res_df)
    else:
        res_list = [cent_pair(cents)[0]]
    print("computing distances")
    res_list = da.compute(res_list)[0]
    res_df = pd.concat(res_list, ignore_index=True)
    res_df.columns = pd.MultiIndex.from_tuples(res_df.columns)
    return res_df


def subset_pairs(A, B, tile):
    Ah, Aw, Bh, Bw = A["height"], A["width"], B["height"], B["width"]
    hh = (min(Ah.min(), Bh.min()), max(Ah.max(), Bh.max()))
    ww = (min(Aw.min(), Bw.min()), max(Aw.max(), Bw.max()))
    dh, dw = int(np.ceil(tile[0] / 2)), int(np.ceil(tile[1] / 2))
    tile_h = np.linspace(hh[0], hh[1], int(np.ceil((hh[1] - hh[0]) * 2 / tile[0])))
    tile_w = np.linspace(ww[0], ww[1], int(np.ceil((ww[1] - ww[0]) * 2 / tile[1])))
    pairs = set()
    for h, w in itt.product(tile_h, tile_w):
        curA = A[Ah.between(h - dh, h + dh) & Aw.between(w - dw, w + dw)]
        curB = B[Bh.between(h - dh, h + dh) & Bw.between(w - dw, w + dw)]
        Au, Bu = curA["unit_id"].values, curB["unit_id"].values
        pairs.update(set(map(tuple, cartesian(Au, Bu).tolist())))
    return pairs


def pd_dist(A, B):
    return np.sqrt(
        ((A[["height", "width"]] - B[["height", "width"]]) ** 2).sum("columns")
    )


def cartesian(*args):
    n = len(args)
    return np.array(np.meshgrid(*args)).T.reshape((-1, n))


def group_by_session(df):
    ss = df["session"].notnull()
    grp = ss.apply(lambda r: tuple(r.index[r].tolist()), axis=1)
    df["group", "group"] = grp
    return df


def calculate_mapping(dist):
    map_idxs = set()
    try:
        for anm, grp in dist.groupby(dist["meta", "animal"]):
            map_idxs.update(mapping(grp))
    except KeyError:
        map_idxs = mapping(dist)
    return dist.loc[list(map_idxs)]


def mapping(dist):
    map_list = set()
    for sess, grp in dist.groupby(dist["group", "group"]):
        minidx_list = []
        for ss in sess:
            minidx = set()
            for uid, uid_grp in grp.groupby(grp["session", ss]):
                minidx.add(uid_grp["variable", "distance"].idxmin())
            minidx_list.append(minidx)
        minidxs = set.intersection(*minidx_list)
        map_list.update(minidxs)
    return map_list


def resolve_mapping(mapping):
    map_list = []
    try:
        for anm, grp in mapping.groupby(mapping["meta", "animal"]):
            map_list.append(resolve(grp))
    except KeyError:
        map_list = [resolve(mapping)]
    return pd.concat(map_list, ignore_index=True)


def resolve(mapping):
    mapping = mapping.reset_index(drop=True)
    map_ss = mapping["session"]
    for ss in map_ss.columns:
        del_idx = []
        for ss_uid, ss_grp in mapping.groupby(mapping["session", ss]):
            if ss_grp.shape[0] > 1:
                del_idx.extend(ss_grp.index)
                new_sess = []
                for s in ss_grp["session"]:
                    uval = ss_grp["session", s].dropna().unique()
                    if len(uval) == 0:
                        new_sess.append(np.nan)
                    elif len(uval) == 1:
                        new_sess.append(uval[0])
                    elif len(uval) > 1:
                        break
                else:
                    new_row = ss_grp.iloc[0].copy()
                    new_row["session"] = new_sess
                    new_row["variable", "distance"] = np.nan
                    mapping = mapping.append(new_row, ignore_index=True)
        mapping = mapping.drop(del_idx).reset_index(drop=True)
    return group_by_session(mapping)


def fill_mapping(mappings, cents):
    def fill(cur_grp, cur_cent):
        fill_ls = []
        for cur_ss in list(cur_grp["session"]):
            cur_ss_grp = cur_grp["session"][cur_ss].dropna()
            cur_ss_all = cur_cent[cur_cent["session"] == cur_ss]["unit_id"].dropna()
            cur_fill_set = set(cur_ss_all.unique()) - set(cur_ss_grp.unique())
            cur_fill_df = pd.DataFrame({("session", cur_ss): list(cur_fill_set),})
            cur_fill_df[("group", "group")] = [(cur_ss,)] * len(cur_fill_df)
            fill_ls.append(cur_fill_df)
        return pd.concat(fill_ls, ignore_index=True)

    meta_cols = list(filter(lambda c: c[0] == "meta", mappings.columns))
    if meta_cols:
        meta_cols_smp = [c[1] for c in meta_cols]
        for cur_id, cur_grp in mappings.groupby(meta_cols):
            cur_cent = cents.set_index(meta_cols_smp).loc[cur_id].reset_index()
            cur_grp_fill = fill(cur_grp, cur_cent)
            cur_id = cur_id if type(cur_id) is tuple else tuple([cur_id])
            for icol, col in enumerate(meta_cols):
                cur_grp_fill[col] = cur_id[icol]
            mappings = pd.concat([mappings, cur_grp_fill], ignore_index=True)
    else:
        map_fill = fill(mappings, cents)
        mappings = pd.concat([mappings, map_fill], ignore_index=True)
    return mappings
