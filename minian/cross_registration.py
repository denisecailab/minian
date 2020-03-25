import os
import re
import itertools as itt
import numpy as np
import xarray as xr
import pandas as pd
import dask as da
from dask.diagnostics import ProgressBar
from scipy.ndimage.measurements import center_of_mass
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from .preprocessing import remove_background
from .motion_correction import estimate_shift_fft, apply_shifts
from .utilities import xrconcat_recursive
from .visualization import centroid
from IPython.core.debugger import set_trace


def load_cnm_dataset_mf(path, pattern=r'^cnm.nc$', concat_dim='session'):
    path = os.path.normpath(path)
    cnmlist = []
    for dirpath, dirnames, fnames in os.walk(path):
        cnmnames = filter(lambda fn: re.search(pattern, fn), fnames)
        cnmpath = [os.path.join(dirpath, cnm) for cnm in cnmnames]
        cnmlist += cnmpath
    if len(cnmlist) > 1:
        return xr.open_mfdataset(cnmlist, concat_dim=concat_dim)
    else:
        print("No CNMF dataset found under path: {}".format(path))
        return None


def load_cnm_dataset(path, pattern=r'^cnm.nc$', concat_dim='session'):
    path = os.path.normpath(path)
    cnmlist = []
    for dirpath, dirnames, fnames in os.walk(path):
        cnmnames = filter(lambda fn: re.search(pattern, fn), fnames)
        for cnm in cnmnames:
            cnmpath = os.path.join(dirpath, cnm)
            cnmds = xr.open_dataset(cnmpath, chunks={})
            cnmds = cnmds.assign_coords(
                animal=cnmds.coords['animal'].astype(str))
            cnmds = cnmds.assign_coords(
                session=cnmds.coords['session'].astype(str))
            cnmds = cnmds.assign_coords(
                session_id=cnmds.coords['session_id'].astype(str))
            cnmds = cnmds.sel(unit_id=cnmds.attrs['unit_mask'])
            cnmlist.append(cnmds)
    if cnmlist:
        return xr.concat(cnmlist, dim=concat_dim)
    else:
        print("No CNMF dataset found under path: {}".format(path))
        return None


def get_minian_list(path, pattern=r'^minian.nc$'):
    path = os.path.normpath(path)
    mnlist = []
    for dirpath, dirnames, fnames in os.walk(path):
        mnames = filter(lambda fn: re.search(pattern, fn), fnames)
        mn_paths = [os.path.join(dirpath, mn) for mn in mnames]
        mnlist += mn_paths
    return mnlist

def estimate_shifts(minian_df, by='session', to='first', temp_var='org', template=None, rm_background=False):
    if template is not None:
        minian_df['template'] = template

    def get_temp(row):
        ds, temp = row['minian'], row['template']
        try:
            return ds.isel(frame=temp).drop('frame')
        except TypeError:
            func_dict = {
                'mean': lambda v: v.mean('frame'),
                'max': lambda v: v.max('frame')}
            try:
                return func_dict[temp](ds)
            except KeyError:
                raise NotImplementedError(
                    "template {} not understood".format(temp))

    minian_df['template'] = minian_df.apply(get_temp, axis='columns')
    grp_dims = list(minian_df.index.names)
    grp_dims.remove(by)
    temp_dict, shift_dict, corr_dict, tempsh_dict = [dict() for _ in range(4)]
    for idxs, df in minian_df.groupby(level=grp_dims):
        try:
            temp_ls = [t[temp_var] for t in df['template']]
        except KeyError:
            raise KeyError(
                "variable {} not found in dataset".format(temp_var))
        temps = (xr.concat(temp_ls, dim=by).expand_dims(grp_dims)
                 .reset_coords(drop=True))
        res = estimate_shift_fft(temps, dim=by, on=to)
        shifts = res.sel(variable=['height', 'width'])
        corrs = res.sel(variable='corr')
        temps_sh = apply_shifts(temps, shifts)
        temp_dict[idxs] = temps
        shift_dict[idxs] = shifts
        corr_dict[idxs] = corrs
        tempsh_dict[idxs] = temps_sh
    temps = xrconcat_recursive(temp_dict, grp_dims).rename('temps')
    shifts = xrconcat_recursive(shift_dict, grp_dims).rename('shifts')
    corrs = xrconcat_recursive(corr_dict, grp_dims).rename('corrs')
    temps_sh = xrconcat_recursive(tempsh_dict, grp_dims).rename('temps_shifted')
    with ProgressBar():
        temps = temps.compute()
        shifts = shifts.compute()
        corrs = corrs.compute()
        temps_sh = temps_sh.compute()
    return xr.merge([temps, shifts, corrs, temps_sh])


def estimate_shifts_old(mn_list,
                    temp_list,
                    z_thres=None,
                    rm_background=False,
                    method='first',
                    concat_dim='session'):
    temps = []
    for imn, mn_path in enumerate(mn_list):
        print(
            "loading template: {:2d}/{:2d}".format(imn, len(mn_list)))
        try:
            with xr.open_dataset(
                mn_path, chunks=dict(width='auto', height='auto'))['org'] as cur_va:
                if temp_list[imn] == 'first':
                    cur_temp = cur_va.isel(frame=0).load().copy()
                elif temp_list[imn] == 'last':
                    cur_temp = cur_va.isel(frame=-1).load().copy()
                elif temp_list[imn] == 'mean':
                    cur_temp = (cur_va.mean('frame'))
                    with ProgressBar():
                        cur_temp = cur_temp.compute()
                else:
                    print("unrecognized template")
                    continue
                if rm_background:
                    cur_temp = remove_background(cur_temp, 'uniform', wnd=51)
                temps.append(cur_temp)
        except KeyError:
            print("no video found for path {}".format(mn_path))
    if concat_dim:
        temps = xr.concat(temps, dim=concat_dim).rename('temps')
        window = ~temps.isnull().sum(concat_dim).astype(bool)
        temps = temps.where(window, drop=True)
    shifts = []
    corrs = []
    for itemp, temp_dst in temps.rolling(**{concat_dim: 1}):
        print("processing: {}".format(itemp.values))
        if method == 'first':
            temp_src = temps.isel(**{concat_dim: 0})
        elif method == 'last':
            temp_src = temps.isel(**{concat_dim: -1})
        # common = (temp_src.isnull() + temp_dst.isnull())
        # temp_src = temp_src.reindex_like(common)
        # temp_dst = temp_dst.reindex_like(common)
        temp_src, temp_dst = temp_src.squeeze(), temp_dst.squeeze()
        src_fft = np.fft.fft2(temp_src)
        dst_fft = np.fft.fft2(temp_dst)
        cur_res = shift_fft(src_fft, dst_fft)
        cur_sh = cur_res[0:2]
        cur_cor = cur_res[2]
        cur_anm = temp_dst.coords['animal']
        cur_ss = temp_dst.coords['session']
        cur_ssid = temp_dst.coords['session_id']
        cur_sh = xr.DataArray(
            cur_sh,
            coords=dict(shift_dim=list(temp_dst.dims)),
            dims=['shift_dim'])
        cur_cor = xr.DataArray(cur_cor)
        cur_sh = cur_sh.assign_coords(
            animal=cur_anm, session=cur_ss, session_id=cur_ssid)
        cur_cor = cur_cor.assign_coords(
            animal=cur_anm, session=cur_ss, session_id=cur_ssid)
        shifts.append(cur_sh)
        corrs.append(cur_cor)
    if concat_dim:
        shifts = xr.concat(shifts, dim=concat_dim).rename('shifts')
        corrs = xr.concat(corrs, dim=concat_dim).rename('corrs')
        temps = xr.concat(temps, dim=concat_dim).rename('temps')
    return shifts, corrs, temps


def apply_shifts_old(var, shifts, inplace=False, dim='session'):
    shifts = shifts.dropna(dim)
    var_list = []
    for dim_n, sh in shifts.groupby(dim):
        sh_dict = (sh.astype(int).to_series().reset_index()
                   .set_index('shift_dim')['shifts'].to_dict())
        var_list.append((var.sel(**{dim: dim_n})
                         .shift(**sh_dict).rename(var.name + "_shifted")))
    return xr.concat(var_list, dim=dim)

def calculate_centroids(A, window):
    A = A.where(window, 0)
    return centroid(A, verbose=True)


def calculate_centroids_old(cnmds, window, grp_dim=['animal', 'session']):
    print("computing centroids")
    cnt_list = []
    for anm, cur_anm in cnmds.groupby('animal'):
        for ss, cur_ss in cur_anm.groupby('session'):
            # cnt = centroids(cur_ss['A_shifted'], window.sel(animal=anm))
            cnt = da.delayed(centroids)(
                cur_ss['A_shifted'], window.sel(animal=anm))
            cnt_list.append(cnt)
    with ProgressBar():
        cnt_list, = da.compute(cnt_list)
    cnts_ds = pd.concat(cnt_list, ignore_index=True)
    cnts_ds.height = cnts_ds.height.astype(float)
    cnts_ds.width = cnts_ds.width.astype(float)
    cnts_ds.unit_id = cnts_ds.unit_id.astype(int)
    cnts_ds.animal = cnts_ds.animal.astype(str)
    cnts_ds.session = cnts_ds.session.astype(str)
    cnts_ds.session_id = cnts_ds.session_id.astype(str)
    return cnts_ds


def centroids(A, window=None):
    A = A.load().dropna('unit_id', how='all')
    if not A.size > 0:
        return pd.DataFrame()
    if window is None:
        window = A.isnull().sum('unit_id') == 0
    try:
        A = A.where(window, drop=True)
    except:
        set_trace()
    A = A.fillna(0)
    meta_dims = set(A.coords.keys()) - set(A.dims)
    meta_dict = {dim: A.coords[dim].values for dim in meta_dims}
    cur_meta = pd.Series(meta_dict)
    cts_list = []
    for uid, cur_uA in A.groupby('unit_id'):
        cur_A = cur_uA.values
        if not (cur_A > 0).any():
            continue
        cur_idxs = cur_uA.dims
        cur_cts = center_of_mass(cur_A)
        cur_cts = pd.Series(cur_cts, index=cur_idxs)
        cur_cts = cur_cts.append(pd.Series(dict(unit_id=uid)))
        cur_cts = cur_cts.append(cur_meta)
        cts_list.append(cur_cts)
    try:
        cts_df = pd.concat(cts_list, axis=1, ignore_index=True).T
    except ValueError:
        cts_df = pd.DataFrame()
    return cts_df


def calculate_centroid_distance(cents, by='session', index_dim=['animal'], tile=(50, 50)):
    res_list = []

    def cent_pair(grp):
        dist_df_ls = []
        len_df = 0
        for (byA, grpA), (byB, grpB) in itt.combinations(list(grp.groupby(by)), 2):
            cur_pairs = subset_pairs(grpA, grpB, tile)
            pairs_ls = list(cur_pairs)
            len_df = len_df + len(pairs_ls)
            subA = (grpA.set_index('unit_id')
                    .loc[[p[0] for p in pairs_ls]]
                    .reset_index())
            subB = (grpB.set_index('unit_id')
                    .loc[[p[1] for p in pairs_ls]]
                    .reset_index())
            dist = da.delayed(pd_dist)(subA, subB).rename('distance')
            dist_df = da.delayed(pd.concat)(
                [subA['unit_id'].rename(byA), subB['unit_id'].rename(byB), dist], axis='columns')
            dist_df = dist_df.rename(columns={
                'distance': ('variable', 'distance'),
                byA: (by, byA),
                byB: (by, byB)})
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
                [pd.Series([idx] * len_df, name=('meta', dim))
                 for idx, dim in zip(idxs, index_dim)],
                axis='columns')
            res_df = da.delayed(pd.concat)([meta_df, dist_df], axis='columns')
            res_list.append(res_df)
    else:
        res_list = [cent_pair(cents)[0]]
    print("computing distances")
    res_list = da.compute(res_list)[0]
    res_df = pd.concat(res_list, ignore_index=True)
    res_df.columns = pd.MultiIndex.from_tuples(res_df.columns)
    return res_df


def subset_pairs(A, B, tile):
    Ah, Aw, Bh, Bw = A['height'], A['width'], B['height'], B['width']
    hh = (min(Ah.min(), Bh.min()), max(Ah.max(), Bh.max()))
    ww = (min(Aw.min(), Bw.min()), max(Aw.max(), Bw.max()))
    dh, dw = np.ceil(tile[0] / 2), np.ceil(tile[1] / 2)
    tile_h = np.linspace(hh[0], hh[1], np.ceil((hh[1] - hh[0]) * 2 / tile[0]))
    tile_w = np.linspace(ww[0], ww[1], np.ceil((ww[1] - ww[0]) * 2 / tile[1]))
    pairs = set()
    for h, w in itt.product(tile_h, tile_w):
        curA = A[
            Ah.between(h - dh, h + dh)
            & Aw.between(w - dw, w + dw)]
        curB = B[
            Bh.between(h - dh, h + dh)
            & Bw.between(w - dw, w + dw)]
        Au, Bu = curA['unit_id'].values, curB['unit_id'].values
        pairs.update(
            set(map(tuple, cartesian(Au, Bu).tolist())))
    return pairs


def pd_dist(A, B):
    return np.sqrt(
        ((A[['height', 'width']] - B[['height', 'width']])**2)
        .sum('columns'))

def cartesian(*args):
    n = len(args)
    return np.array(np.meshgrid(*args)).T.reshape((-1, n))


def calculate_centroid_distance_old(cents,
                                A,
                                window,
                                grp_dim=['animal'],
                                tile=(50, 50),
                                shift=True,
                                hamming=True,
                                corr=False):
    dist_list = []
    A = da.delayed(A)
    for cur_anm, cur_grp in cents.groupby('animal'):
        print("processing animal: {}".format(cur_anm))
        cur_A = A.sel(animal=cur_anm)
        cur_wnd = window.sel(animal=cur_anm)
        dist = centroids_distance(cur_grp, cur_A, cur_wnd, shift, hamming,
                                  corr, tile)
        dist['meta', 'animal'] = cur_anm
        dist_list.append(dist)
    dist = pd.concat(dist_list, ignore_index=True)
    return dist


def centroids_distance_old(cents,
                       A,
                       window,
                       shift,
                       hamming,
                       corr,
                       tile=(50, 50)):
    sessions = cents['session'].unique()
    dim_h = (np.min(cents['height']), np.max(cents['height']))
    dim_w = (np.min(cents['width']), np.max(cents['width']))
    dist_list = []
    for ssA, ssB in itt.combinations(sessions, 2):
        # dist = _calc_cent_dist(ssA, ssB, cents, cnmds, window, tile, dim_h, dim_w)
        dist = da.delayed(_calc_cent_dist)(ssA, ssB, cents, A, window,
                                           tile, dim_h, dim_w, shift, hamming,
                                           corr)
        dist_list.append(dist)
    with ProgressBar():
        dist_list, = da.compute(dist_list)
    dists = pd.concat(dist_list, ignore_index=True)
    return dists


def _calc_cent_dist_old(ssA, ssB, cents, A, window, tile, dim_h, dim_w, shift,
                    hamming, corr):
    ssA_df = cents[cents['session'] == ssA]
    ssB_df = cents[cents['session'] == ssB]
    ssA_uids = ssA_df['unit_id'].unique()
    ssB_uids = ssB_df['unit_id'].unique()
    ssA_h = ssA_df['height']
    ssA_w = ssA_df['width']
    ssB_h = ssB_df['height']
    ssB_w = ssB_df['width']
    tile_ct_h = np.linspace(dim_h[0], dim_h[1],
                            np.ceil((dim_h[1] - dim_h[0]) * 2.0 / tile[0]) + 1)
    tile_ct_w = np.linspace(dim_w[0], dim_w[1],
                            np.ceil((dim_w[1] - dim_w[0]) * 2.0 / tile[1]) + 1)
    dh = np.ceil(tile[0] / 2.0)
    dw = np.ceil(tile[1] / 2.0)
    pairs = set()
    for ct_h, ct_w in itt.product(tile_ct_h, tile_ct_w):
        ssA_uid_inrange = ssA_uids[(ct_h - dh < ssA_h)
                                   & (ssA_h < ct_h + dh)
                                   & (ct_w - dw < ssA_w) & (ssA_w < ct_w + dw)]
        ssB_uid_inrange = ssB_uids[(ct_h - dh < ssB_h)
                                   & (ssB_h < ct_h + dh)
                                   & (ct_w - dw < ssB_w) & (ssB_w < ct_w + dw)]
        for pair in itt.product(ssA_uid_inrange, ssB_uid_inrange):
            pairs.add(pair)
    dist_list = []
    for ip, (uidA, uidB) in enumerate(pairs):
        idxarr = [[
            'session', 'session', 'variable', 'variable', 'variable',
            'variable'
        ], [ssA, ssB, 'distance', 'coeff', 'p', 'hamming']]
        mulidx = pd.MultiIndex.from_arrays(
            idxarr, names=('var_class', 'var_name'))
        centA = ssA_df[ssA_df['unit_id'] == uidA][['height', 'width']]
        centB = ssB_df[ssB_df['unit_id'] == uidB][['height', 'width']]
        diff = centA.reset_index(drop=True) - centB.reset_index(drop=True)
        diff = diff.T.squeeze()
        cur_dist = np.sqrt((diff**2).sum())
        if corr or hamming:
            cur_A_A = A.sel(
                session=ssA, unit_id=uidA).where(
                    window, drop=True)
            cur_A_B = A.sel(
                session=ssB, unit_id=uidB).where(
                    window, drop=True)
        if shift:
            cur_A_B = cur_A_B.shift(**diff.round().astype(int).to_dict())
            # wnd_new = cur_A_B.notnull()
            wnd_new = (cur_A_B + cur_A_B) > 0
            cur_A_A = cur_A_A.where(wnd_new, drop=True).fillna(0)
            cur_A_B = cur_A_B.where(wnd_new, drop=True).fillna(0)
        if corr:
            cur_coef, cur_p = pearsonr(cur_A_A.values.flatten(),
                                       cur_A_B.values.flatten())
        else:
            cur_coef, cur_p = np.nan, np.nan
        if hamming:
            ham = xr.apply_ufunc(
                np.absolute, (cur_A_A > 0) - (cur_A_B > 0),
                dask='allowed').sum()
            uni = ((cur_A_A + cur_A_B) > 0).sum()
            ham = np.asscalar((ham / uni).values)
        else:
            ham = np.nan
        dist = pd.Series(
            [uidA, uidB, cur_dist, cur_coef, cur_p, ham], index=mulidx)
        dist_list.append(dist)
    dists = pd.concat(dist_list, axis=1, ignore_index=True).T
    return dists


def group_by_session(df):
    ss = df['session'].notnull()
    grp = ss.apply(lambda r: tuple(r.index[r].tolist()), axis=1)
    df['group', 'group'] = grp
    return df


def calculate_mapping(dist):
    map_idxs = set()
    try:
        for anm, grp in dist.groupby(dist['meta', 'animal']):
            map_idxs.update(mapping(grp))
    except KeyError:
        map_idxs = mapping(dist)
    return dist.loc[list(map_idxs)]


def mapping(dist):
    map_list = set()
    for sess, grp in dist.groupby(dist['group', 'group']):
        minidx_list = []
        for ss in sess:
            minidx = set()
            for uid, uid_grp in grp.groupby(grp['session', ss]):
                minidx.add(uid_grp['variable', 'distance'].idxmin())
            minidx_list.append(minidx)
        minidxs = set.intersection(*minidx_list)
        map_list.update(minidxs)
    return map_list


def resolve_mapping(mapping):
    map_list = []
    try:
        for anm, grp in mapping.groupby(mapping['meta', 'animal']):
            map_list.append(resolve(grp))
    except KeyError:
        map_list = [resolve(mapping)]
    return pd.concat(map_list, ignore_index=True)


def resolve(mapping):
    mapping = mapping.reset_index(drop=True)
    map_ss = mapping['session']
    for ss in map_ss.columns:
        del_idx = []
        for ss_uid, ss_grp in mapping.groupby(mapping['session', ss]):
            if ss_grp.shape[0] > 1:
                del_idx.extend(ss_grp.index)
                new_sess = []
                for s in ss_grp['session']:
                    uval = ss_grp['session', s].dropna().unique()
                    if len(uval) == 0:
                        new_sess.append(np.nan)
                    elif len(uval) == 1:
                        new_sess.append(uval[0])
                    elif len(uval) > 1:
                        break
                else:
                    new_row = ss_grp.iloc[0].copy()
                    new_row['session'] = new_sess
                    new_row['variable', 'distance'] = np.nan
                    mapping = mapping.append(new_row, ignore_index=True)
        mapping = mapping.drop(del_idx).reset_index(drop=True)
    return group_by_session(mapping)


def fill_mapping(mappings,
                 cents):

    def fill(cur_grp, cur_cent):
        fill_ls = []
        for cur_ss in list(cur_grp['session']):
            cur_ss_grp = cur_grp['session'][cur_ss].dropna()
            cur_ss_all = cur_cent[cur_cent['session'] == cur_ss][
                'unit_id'].dropna()
            cur_fill_set = set(cur_ss_all.unique()) - set(
                cur_ss_grp.unique())
            cur_fill_df = pd.DataFrame({
                ('session', cur_ss):
                list(cur_fill_set),
            })
            fill_ls.append(cur_fill_df)
        return pd.concat(fill_ls, ignore_index=True)

    try:
        for cur_id, cur_grp in mappings.groupby(list(mappings['meta'])):
            cur_cent = (cents.set_index(list(mappings['meta']))
                        .loc[cur_id].reset_index())
            cur_grp_fill = fill(cur_grp, cur_cent)
            mappings = pd.concat([mappings, cur_grp_fill], ignore_index=True)
    except KeyError:
        map_fill = fill(mappings, cents)
        mappings = pd.concat([mappings, map_fill], ignore_index=True)
    return mappings
