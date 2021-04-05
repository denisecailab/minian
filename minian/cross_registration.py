import itertools as itt

import dask as da
import numpy as np
import pandas as pd
import xarray as xr
from typing import Iterable

from .visualization import centroid


def calculate_centroids(A: xr.DataArray, window: xr.DataArray) -> pd.DataFrame:
    """
    Calculate centroids of spatial footprints for cells inside a window.

    Parameters
    ----------
    A : xr.DataArray
        The input spatial footprints of cells.
    window : xr.DataArray
        Boolean mask with dimensions "height" and "width". Only sptial
        footprints of cells within this window will be included in the result.

    Returns
    -------
    cents : pd.DataFrame
        Resulting centroids dataframe.

    See Also
    --------
    minian.visualization.centroid
    """
    A = A.where(window, 0)
    return centroid(A, verbose=True)


def calculate_centroid_distance(
    cents: pd.DataFrame, by="session", index_dim=["animal"], tile=(50, 50)
) -> pd.DataFrame:
    """
    Calculate pairwise distance between centroids across all pairs of sessions.

    To avoid calculating distance between centroids that are very far away, a 2d
    rolling window is applied to spatial coordinates, and only pairs of
    centroids within the rolling windows are considered for calculation.

    Parameters
    ----------
    cents : pd.DataFrame
        Dataframe of centroid locations as returned by
        :func:`calculate_centroids`.
    by : str, optional
        Name of column by which cells from sessions will be grouped together. By
        default `"session"`.
    index_dim : list, optional
        Additional metadata columns by which data should be grouped together.
        Pairs of sessions within such groups (but not across groups) will be
        used for calculation. By default `["animal"]`.
    tile : tuple, optional
        Size of the rolling window to constrain caculation, specified in pixels
        and in the order ("height", "width"). By default `(50, 50)`.

    Returns
    -------
    res_df : pd.DataFrame
        Pairwise distance between centroids across all pairs of sessions, where
        each row represent a specific pair of cells across specific sessions.
        The dataframe contains a two-level :doc:`MultiIndex
        <pandas:user_guide/advanced>` as column names. The top level contains
        three labels: "session", "variable" and "meta". Each session will have a
        column under the "session" label, with values indicating the "unit_id"
        of the cell pair if either cell is in the corresponding session, and
        `NaN` otherwise. "variable" contains a single column "distance"
        indicating the distance of centroids for the cell pair. "meta" contains
        all additional metadata dimensions specified in `index_dim` as columns
        so that cell pairs can be uniquely identified.
    """
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


def subset_pairs(A: pd.DataFrame, B: pd.DataFrame, tile: tuple) -> set:
    """
    Return all pairs of cells within certain window given two sets of centroid
    locations.

    Parameters
    ----------
    A : pd.DataFrame
        Input centroid locations. Should have columns "height" and "width".
    B : pd.DataFrame
        Input centroid locations. Should have columns "height" and "width".
    tile : tuple
        Window size.

    Returns
    -------
    pairs : set
        Set of all cell pairs represented as tuple.
    """
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


def pd_dist(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
    """
    Compute euclidean distance between two sets of matching centroid locations.

    Parameters
    ----------
    A : pd.DataFrame
        Input centroid locations. Should have columns "height" and "width".
    B : pd.DataFrame
        Input centroid locations. Should have columns "height" and "width" and
        same row index as `A`, such that distance between corresponding rows
        will be calculated.

    Returns
    -------
    dist : pd.Series
        Distance between centroid locations. Has same row index as `A` and `B`.
    """
    return np.sqrt(
        ((A[["height", "width"]] - B[["height", "width"]]) ** 2).sum("columns")
    )


def cartesian(*args: Iterable) -> np.ndarray:
    """
    Computes cartesian product of inputs.

    Parameters
    ----------
    *args : array_like
        Inputs that can be interpreted as array.

    Returns
    -------
    product : np.ndarray
        k x n array representing cartesian product of inputs, with k number of
        unique combinations for n inputs.
    """
    n = len(args)
    return np.array(np.meshgrid(*args)).T.reshape((-1, n))


def group_by_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add grouping information based on sessions involved in each row/mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with rows representing mappings. Should be in two-level
        column format like those returned by :func:`calculate_centroid_distance`
        or :func:`calculate_mapping` etc.

    Returns
    -------
    df : pd.DataFrame
        The input `df` with an additional ("group", "group") column, whose
        values are tuples indicating which sessions are involved (have non-NaN
        values) in the mappings represented by each row.

    See Also
    --------
    resolve_mapping : for example usages
    """
    ss = df["session"].notnull()
    grp = ss.apply(lambda r: tuple(r.index[r].tolist()), axis=1)
    df["group", "group"] = grp
    return df


def calculate_mapping(dist: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mappings from cell pair distances with mutual nearest-neighbor
    criteria.

    This function takes in distance between cell pairs and filter them based on
    mutual nearest-neighbor criteria, where a cell pair is considered a valid
    mapping only when either cell is the nearest neighbor to the other (among
    all cell pairs presented in input `dist`). The result is hence a subset of
    input `dist` dataframe and rows are considered mapping between cells in
    pairs of sessions.

    Parameters
    ----------
    dist : pd.DataFrame
        The distances between cell pairs. Should be in two-level column format
        as returned by :func:`calculate_centroid_distance`, and should also
        contains a ("group", "group") column as returned by
        :func:`group_by_session`.

    Returns
    -------
    mapping : pd.DataFrame
        The mapping of cells across sessions, where each row represent a mapping
        of cells across specific sessions. The dataframe contains a two-level
        :doc:`MultiIndex <pandas:user_guide/advanced>` as column names. The top
        level contains three labels: "session", "variable" and "meta". Each
        session will have a column under the "session" label, with values
        indicating the "unit_id" of the cell in that session involved in the
        mapping, or `NaN` if the mapping does not involve the session.
        "variable" contains a single column "distance" indicating the distance
        of centroids for the cell pair if the mapping involve only two cells,
        and `NaN` otherwise. "meta" contains all additional metadata dimensions
        specified in `index_dim` as columns so that cell pairs can be uniquely
        identified.
    """
    map_idxs = set()
    meta_cols = list(filter(lambda c: c[0] == "meta", dist.columns))
    if meta_cols:
        for _, grp in dist.groupby(meta_cols):
            map_idxs.update(cal_mapping(grp))
    else:
        map_idxs = cal_mapping(dist)
    return dist.loc[list(map_idxs)]


def cal_mapping(dist: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mappings from cell pair distances for a single group.

    This function is called by :func:`calculate_mapping` for each group defined
    by metadata.

    Parameters
    ----------
    dist : pd.DataFrame
        The distances between cell pairs. Should be in two-level column format.

    Returns
    -------
    mapping : pd.DataFrame
        The mapping of cells across sessions.

    See Also
    --------
    calculate_mapping
    """
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


def resolve_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Extend and resolve mappings of pairs of sessions into mappings across
    multiple sessions.

    This function try to transitively extend any mappings that share common
    cells. If such mappings are consistent with each other, they will be merged
    into a single mapping that maps a single cell across all sessions. Otherwise
    if any of the mappings introduce conflicts, then all of the relevant
    mappings will be dropped.

    Parameters
    ----------
    mapping : pd.DataFrame
        Input mappings dataframe. Should be in two-level column format as
        returned by :func:`calculate_mapping`, and should also contains a
        ("group", "group") column as returned by :func:`group_by_session`.

    Returns
    -------
    mapping : pd.DataFrame
        Output mappings with extended and resolved mappings. Should be in the
        same two-level column format as input.

    Examples
    --------
    Suppose we have two mappings sharing a common cell in "session2":

    >>> mapping = pd.DataFrame(
    ...     {
    ...         ("meta", "animal"): ["m1", "m1"],
    ...         ("session", "session1"): [0, None],
    ...         ("session", "session2"): [1, 1],
    ...         ("session", "session3"): [None, 2],
    ...     }
    ... )
    >>> mapping = group_by_session(mapping)
    >>> mapping # doctest: +NORMALIZE_WHITESPACE
        meta  session                                   group
      animal session1 session2 session3                 group
    0     m1      0.0        1      NaN  (session1, session2)
    1     m1      NaN        1      2.0  (session2, session3)

    Then they will be extended and merged as a single mapping:

    >>> resolve_mapping(mapping) # doctest: +NORMALIZE_WHITESPACE
        meta  session                                             group variable
      animal session1 session2 session3                           group distance
    0     m1      0.0        1      2.0  (session1, session2, session3)      NaN

    However, if our mappings contains an additional entry that conflicts with
    the extended mapping like the following:

    >>> mapping = pd.DataFrame(
    ...     {
    ...         ("meta", "animal"): ["m1", "m1", "m1"],
    ...         ("session", "session1"): [0, None, 0],
    ...         ("session", "session2"): [1, 1, None],
    ...         ("session", "session3"): [None, 2, 5],
    ...     }
    ... )
    >>> mapping = group_by_session(mapping)
    >>> mapping # doctest: +NORMALIZE_WHITESPACE
        meta  session                                   group
      animal session1 session2 session3                 group
    0     m1      0.0      1.0      NaN  (session1, session2)
    1     m1      NaN      1.0      2.0  (session2, session3)
    2     m1      0.0      NaN      5.0  (session1, session3)

    Then all of them will be dropped:

    >>> resolve_mapping(mapping) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Empty DataFrame
    Columns: [(meta, animal), ... (variable, distance)]
    Index: []
    """
    map_list = []
    meta_cols = list(filter(lambda c: c[0] == "meta", mapping.columns))
    if meta_cols:
        for _, grp in mapping.groupby(meta_cols):
            map_list.append(resolve(grp))
    else:
        map_list = [resolve(mapping)]
    return pd.concat(map_list, ignore_index=True)


def resolve(mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Extend and resolve mappings.

    This function is called by :func:`resolve_mapping` for each group defined by
    metadata

    Parameters
    ----------
    mapping : pd.DataFrame
        Input mappings dataframe. Should be in two-level column format.

    Returns
    -------
    mapping : pd.DataFrame
        Output mappings with extended and resolved mappings. Should be in the
        same two-level column format as input.

    See Also
    --------
    resolve_mapping
    """
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


def fill_mapping(mappings: pd.DataFrame, cents: pd.DataFrame) -> pd.DataFrame:
    """
    Fill mappings with rows representing unmatched cells.

    This function takes all cells in `cents` and check to see if they appear in
    any rows in `mappings`. If a cell is not involved in any mappings, then a
    row will be appended to `mappings` with the cell's "unit_id" in the session
    column contatining the cell and `NaN` in all other "session" columns.

    Parameters
    ----------
    mappings : pd.DataFrame
        Input mappings dataframe. Should be in two-level column format as
        returned by :func:`calculate_mapping`, and should also contains a
        ("group", "group") column as returned by :func:`group_by_session`.
    cents : pd.DataFrame
        Dataframe of centroid locations as returned by
        :func:`calculate_centroids`.

    Returns
    -------
    mappings : pd.DataFrame
        Output mappings with unmatched cells.
    """

    def fill(cur_grp, cur_cent):
        fill_ls = []
        for cur_ss in list(cur_grp["session"]):
            cur_ss_grp = cur_grp["session"][cur_ss].dropna()
            cur_ss_all = cur_cent[cur_cent["session"] == cur_ss]["unit_id"].dropna()
            cur_fill_set = set(cur_ss_all.unique()) - set(cur_ss_grp.unique())
            cur_fill_df = pd.DataFrame({("session", cur_ss): list(cur_fill_set)})
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
