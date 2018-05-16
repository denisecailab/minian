import os
import re
import itertools as itt
import numpy as np
import xarray as xr
import pandas as pd
import dask as da
from dask.diagnostics import ProgressBar
from scipy.ndimage.measurements import center_of_mass
from .motion_correction import shift_fft
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


def get_cnm_list(path, pattern=r'^cnm.nc$'):
    path = os.path.normpath(path)
    cnmlist = []
    for dirpath, dirnames, fnames in os.walk(path):
        cnmnames = filter(lambda fn: re.search(pattern, fn), fnames)
        cnm_paths = [os.path.join(dirpath, cnm) for cnm in cnmnames]
        cnmlist += cnm_paths
    return cnmlist


def estimate_shifts(cnm_list,
                    temp_list,
                    z_thres=None,
                    method='first',
                    concat_dim='session'):
    temps = []
    for icnm, cnm_path in enumerate(cnm_list):
        print(
            "loading template: {:2d}/{:2d}".format(icnm, len(cnm_list)),
            end='\r')
        with xr.open_dataset(cnm_path) as cnm:
            cur_path = os.path.dirname(
                cnm.attrs['file_path']) + os.sep + 'varr_mc_int.nc'
        try:
            with xr.open_dataset(cur_path)['varr_mc_int'] as cur_va:
                if temp_list[icnm] == 'first':
                    cur_temp = cur_va.sel(
                        frame=cur_va.coords['frame'][0]).load().copy()
                    temps.append(cur_temp)
                elif temp_list[icnm] == 'last':
                    cur_temp = cur_va.sel(
                        frame=cur_va.coords['frame'][-1]).load().copy()
                    temps.append(cur_temp)
                elif temp_list[icnm] == 'mean':
                    cur_va = cur_va.load().chunk(dict(width=50, height=50))
                    cur_temp = cur_va.mean('frame').compute()
                    temps.append(cur_temp)
                else:
                    print("unrecognized template")
        except KeyError:
            print("no varr found for path {}".format(cnm_path))
    shifts = []
    corrs = []
    for itemp, temp_dst in enumerate(temps):
        print(
            "estimating shifts: {:2d}/{:2d}".format(itemp, len(temps)),
            end='\r')
        if method == 'first':
            temp_src = temps[0]
        common = (temp_src.isnull() + temp_dst.isnull())
        temp_src = temp_src.reindex_like(common)
        temp_dst = temp_dst.reindex_like(common)
        cur_sh, cur_cor = shift_fft(temp_src, temp_dst)
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


def apply_shifts(var, shifts, inplace=False, dim='session'):
    var_sh = var.astype('O', copy=not inplace)
    for dim_n, sh in shifts.groupby(dim):
        sh_dict = sh.astype(int).to_series().to_dict()
        var_sh.loc[{dim: dim_n}] = var_sh.loc[{dim: dim_n}].shift(**sh_dict)
    return var_sh.rename(var.name + '_shifted')


def calculate_centroids(cnmds, window, grp_dim=['animal', 'session']):
    print("computing centroids")
    cnt_list = []
    for anm, cur_anm in cnmds.groupby('animal'):
        for ss, cur_ss in cur_anm.groupby('session'):
            cnt = da.delayed(centroids)(cur_ss['A_shifted'], window.sel(animal=anm))
            cnt_list.append(cnt)
    with ProgressBar():
        cnt_list, = da.compute(cnt_list)
    cnts_ds = pd.concat(cnt_list, ignore_index=True)
    return cnts_ds


def centroids(A, window=None):
    A = A.load().dropna('unit_id', how='all')
    if window is None:
        window = A.isnull().sum('unit_id') == 0
    A = A.where(window, drop=True)
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


def calculate_centroid_distance(cents, grp_dim=['animal'], tile=(50, 50)):
    dist_list = []
    for cur_anm, cur_grp in cents.groupby('animal'):
        print("processing animal: {}".format(cur_anm))
        dist = centroids_distance(cur_grp, tile)
        dist['meta', 'animal'] = cur_anm
        dist_list.append(dist)
    dist = pd.concat(dist_list, ignore_index=True)
    return dist


def centroids_distance(cents, tile=(50, 50)):
    sessions = cents['session'].unique()
    dim_h = (np.min(cents['height']), np.max(cents['height']))
    dim_w = (np.min(cents['width']), np.max(cents['width']))
    dist_list = []
    for ssA, ssB in itt.combinations(sessions, 2):
        dist = da.delayed(_calc_cent_dist)(ssA, ssB, cents, tile, dim_h, dim_w)
        dist_list.append(dist)
    with ProgressBar():
        dist_list, = da.compute(dist_list)
    dists = pd.concat(dist_list, ignore_index=True)
    return dists


def _calc_cent_dist(ssA, ssB, cents, tile, dim_h, dim_w):
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
        idxarr = [['session', 'session', 'variable'], [ssA, ssB, 'distance']]
        mulidx = pd.MultiIndex.from_arrays(
            idxarr, names=('var_class', 'var_name'))
        centA = ssA_df[ssA_df['unit_id'] == uidA][['height',
                                                   'width']].values.squeeze()
        centB = ssB_df[ssB_df['unit_id'] == uidB][['height',
                                                   'width']].values.squeeze()
        cur_dist = np.sqrt(np.sum((centA - centB)**2))
        dist = pd.Series([uidA, uidB, cur_dist], index=mulidx)
        dist_list.append(dist)
    dists = pd.concat(dist_list, axis=1, ignore_index=True).T
    return dists


def group_by_session(df):
    ss = df['session'].notnull()
    grp = ss.apply(lambda r: tuple(r.index[r].tolist()), axis=1)
    df['meta', 'group'] = grp
    return df


def calculate_mapping(dist):
    map_idxs = set()
    for anm, grp in dist.groupby(dist['meta', 'animal']):
        map_idxs.update(mapping(grp))
    return dist.loc[list(map_idxs)]


def mapping(dist):
    map_list = set()
    for sess, grp in dist.groupby(dist['meta', 'group']):
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
    for anm, grp in mapping.groupby(mapping['meta', 'animal']):
        map_list.append(resolve(grp))
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
