from minian.utilities import save_minian
import numpy as np
from numpy.core.fromnumeric import amin
import xarray as xr
import os
from numpy import random
from ..preprocessing import *
from ..motion_correction import *
from ..initialization import *
from ..cnmf import *
from ..visualization import convolve_G, write_video
from cv2 import GaussianBlur


def gauss_cell(
    height: int, width: int, sigma: float, cov_coef: float, cent=None, nsamp=1000
):
    # generate centroid
    if cent is None:
        cent = (random.randint(height), random.randint(width))
    # generate covariance
    while True:
        cov_var = random.rand(2, 2)
        cov_var = (cov_var + cov_var.T) / 2 * cov_coef
        cov = np.eye(2) * sigma + cov_var
        if np.all(np.linalg.eigvals(cov) > 0):
            break  # ensure cov is positive definite
    # generate samples of coordinates
    crds = np.clip(
        np.round(random.multivariate_normal(cent, cov, size=nsamp)).astype(np.int),
        0,
        None,
    )
    # generate spatial footprint
    A = np.zeros((height, width))
    for crd in np.unique(crds, axis=0):
        try:
            A[tuple(crd)] = np.sum(np.all(crds == crd, axis=1))
        except IndexError:
            pass
    return A / A.max()


def ar_trace(frame: int, pfire: float, g: np.ndarray):
    S = random.binomial(n=1, p=pfire, size=frame)
    C = convolve_G(S, g)
    return C, S


def generate_data(
    ncell: int,
    dims: dict,
    sp_noise: float,
    tmp_noise: float,
    sp_sigma: float,
    sp_cov_coef: float,
    tmp_pfire: float,
    tmp_g_avg: float,
    tmp_g_var: float,
    bg_sigma=0,
    bg_strength=0,
    mo_sigma=0,
):
    ff, hh, ww = (
        dims["frame"],
        dims["height"],
        dims["width"],
    )
    hh_pad, ww_pad = hh + mo_sigma * 4, ww + mo_sigma * 4
    A = xr.DataArray(
        np.stack(
            [
                gauss_cell(hh_pad, ww_pad, sp_sigma, cov_coef=sp_cov_coef)
                for _ in range(ncell)
            ]
        ),
        dims=["unit_id", "height", "width"],
        coords={
            "height": np.arange(hh_pad),
            "width": np.arange(ww_pad),
            "unit_id": np.arange(ncell),
        },
        name="A",
    )
    tmp_g = np.clip(
        random.normal(tmp_g_avg, tmp_g_var, size=ncell), a_min=0.8, a_max=0.95
    )
    traces = [ar_trace(ff, tmp_pfire, np.array([g])) for g in tmp_g]
    C = xr.DataArray(
        np.stack([t[0] for t in traces]),
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(ncell), "frame": np.arange(ff)},
        name="C",
    )
    S = xr.DataArray(
        np.stack([t[1] for t in traces]),
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(ncell), "frame": np.arange(ff)},
        name="S",
    )
    C_noise = C + random.normal(scale=tmp_noise, size=(ncell, ff))
    Y = C_noise.dot(A).rename("Y")
    Y = Y / Y.max()
    if bg_strength:
        A_bg = xr.apply_ufunc(
            GaussianBlur,
            A,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            kwargs={
                "ksize": (int(hh // 2 * 2 - 1), int(ww // 2 * 2 - 1)),
                "sigmaX": bg_sigma,
            },
        )
        Y_bg = C_noise.dot(A_bg)
        Y_bg = Y_bg / Y_bg.max()
        Y = Y + Y_bg * bg_strength
    shifts = xr.DataArray(
        np.clip(
            random.normal(scale=mo_sigma, size=(ff, 2)),
            a_min=-2 * mo_sigma,
            a_max=2 * mo_sigma,
        ),
        dims=["frame", "variable"],
        coords={"frame": np.arange(ff), "variable": ["height", "width"]},
        name="shifts",
    )
    Y = (
        apply_shifts(Y, shifts)
        .compute()
        .isel(
            height=slice(2 * mo_sigma, -2 * mo_sigma),
            width=slice(2 * mo_sigma, -2 * mo_sigma),
        )
    )
    Y = (Y / Y.max() + random.normal(scale=sp_noise, size=(ff, hh, ww))).rename("Y")
    return Y, A, C, S, shifts


if __name__ == "__main__":
    # generate toy data
    testpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "toy")
    Y, A, C, S, shifts = generate_data(
        ncell=100,
        dims={"height": 100, "width": 100, "frame": 1000},
        sp_noise=0.05,
        tmp_noise=0.08,
        sp_sigma=3,
        sp_cov_coef=2,
        tmp_pfire=0.1,
        tmp_g_avg=0.9,
        tmp_g_var=0.03,
        bg_sigma=20,
        bg_strength=1,
        mo_sigma=1,
    )
    Y_true = C.dot(A)
    Y_true = (
        ((Y_true - Y_true.min()) / (Y_true.max() - Y_true.min()) * 255)
        .astype(np.uint8)
        .rename("Y_true")
    )
    Y = (
        ((Y - Y.min()) / (Y.max() - Y.min()) * 255)
        .astype(np.uint8)
        .chunk({"frame": 500})
    )
    write_video(Y, "toy.mp4", testpath)
    # run pipeline
    print("pre-processing")
    Y_glow = (Y - Y.min("frame").compute()).rename("Y_glow")
    Y_dn = denoise(Y_glow, method="median", ksize=5).rename("Y_denoise")
    Y_bg = remove_background(Y_dn, method="tophat", wnd=5).rename("Y_bg")
    shifts_est = estimate_shifts(Y_bg, max_sh=10)
    Y_mc = apply_shifts(Y_bg, shifts_est).fillna(0).rename("Y_mc")
    print("initialization")
    Y_flt = Y.compute().stack(spatial=["height", "width"])
    seeds_in = seeds_init(Y_mc, wnd_size=200, stp_size=100, diff_thres=2)
    seeds_pnr, _, _ = pnr_refine(Y_flt, seeds_in.copy(), noise_freq=0.01, thres=0.5)
    seeds_ks = ks_refine(Y_flt, seeds_pnr[seeds_pnr["mask_pnr"]])
    seeds_mrg = seeds_merge(
        Y_flt,
        seeds_ks[seeds_ks["mask_ks"]].reset_index(drop=True),
        thres_dist=2,
        thres_corr=0.7,
        noise_freq=0.01,
    )
    A_init, C_init, b_init, f_init = initialize(
        Y_mc, seeds_mrg[seeds_mrg["mask_mrg"]], thres_corr=0.8, wnd=15, noise_freq=0.01
    )
    print("cnmf")
    sn = get_noise_fft(Y_mc).persist()
    A_sp1, b_sp1, C_sp1, f_sp1 = update_spatial(
        Y_mc, A_init, b_init, C_init, f_init, sn
    )
    YrA, C_tp1, S_tp1, B_tp1, C0_tp1, sig_tp1, g_tp1, scale_tp1 = update_temporal(
        Y_mc, A_sp1, b_sp1, C_sp1, f_sp1, sn
    )
    A_tp1 = A_sp1.sel(unit_id=C_tp1.coords["unit_id"]).rename("A_tp1")
    A_mrg, sig_mrg = unit_merge(A_tp1, sig_tp1, [])
    print(A_mrg)
    # save results
    (
        A_init,
        C_init,
        b_init,
        f_init,
        sn,
        A_sp1,
        b_sp1,
        C_sp1,
        f_sp1,
        C_tp1,
        S_tp1,
        sig_tp1,
        A_mrg,
        sig_mrg,
    ) = (
        A_init.rename("A_init"),
        C_init.rename("C_init"),
        b_init.rename("b_init"),
        f_init.rename("f_init"),
        sn.rename("sn"),
        A_sp1.rename("A_sp1"),
        b_sp1.rename("b_sp1"),
        C_sp1.rename("C_sp1"),
        f_sp1.rename("f_sp1"),
        C_tp1.rename("C_tp1"),
        S_tp1.rename("S_tp1"),
        sig_tp1.rename("sig_tp1"),
        A_mrg.rename("A_mrg"),
        sig_mrg.rename("sig_mrg"),
    )
    for dat in [
        Y,
        A,
        C,
        S,
        shifts,
        Y_true,
        Y_glow,
        Y_dn,
        Y_bg,
        Y_mc,
        A_init,
        C_init,
        b_init,
        f_init,
        sn,
        A_sp1,
        b_sp1,
        C_sp1,
        f_sp1,
        C_tp1,
        S_tp1,
        sig_tp1,
        A_mrg,
        sig_mrg,
    ]:
        save_minian(
            dat,
            dpath=testpath,
            fname="minian",
            backend="zarr",
            meta_dict={"session": -1, "animal": -2},
            overwrite=True,
        )

