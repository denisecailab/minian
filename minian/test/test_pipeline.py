import pytest

import itertools as itt
import os
import sys
import ffmpeg
import shutil

import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from holoviews.operation.datashader import datashade
from holoviews.util import Dynamic
from IPython.core.display import display

# Set up Initial Basic Parameters#
minian_path = "."
dpath = "./minian/test/test_movie"
dpath_fixture = "./minian/test/test_movie_fixture"
minian_ds_path = os.path.join(dpath, "minian")
intpath = "./minian/test/minian_intermediate"
subset = dict(frame=slice(0, None))
subset_mc = None
interactive = True
output_size = 100
param_save_minian = {
    "dpath": minian_ds_path,
    "meta_dict": dict(session_id=-1, session=-2, animal=-3),
    "overwrite": True,
}

# Pre-processing Parameters#
param_load_videos = {
    "pattern": "msCam[0-9]+\.avi$",
    "dtype": np.uint8,
    "downsample": dict(frame=1, height=1, width=1),
    "downsample_strategy": "subset",
}
param_denoise = {"method": "median", "ksize": 7}
param_background_removal = {"method": "tophat", "wnd": 15}

# Motion Correction Parameters#
subset_mc = None
param_estimate_shift = {"dim": "frame", "max_sh": 20}

# Initialization Parameters#
param_seeds_init = {
    "wnd_size": 1000,
    "method": "rolling",
    "stp_size": 500,
    "nchunk": 100,
    "max_wnd": 15,
    "diff_thres": 2,
}
param_pnr_refine = {"noise_freq": 0.06, "thres": 1, "med_wnd": None}
param_ks_refine = {"sig": 0.05}
param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.7, "noise_freq": 0.06}
param_initialize = {"thres_corr": 0.8, "wnd": 15, "noise_freq": 0.06}

# CNMF Parameters#
param_get_noise = {"noise_range": (0.06, 0.5), "noise_method": "logmexp"}
param_first_spatial = {
    "dl_wnd": 5,
    "sparse_penal": 0.01,
    "update_background": True,
    "normalize": True,
    "size_thres": (25, None),
}
param_first_temporal = {
    "noise_freq": 0.06,
    "sparse_penal": 1,
    "p": 1,
    "add_lag": 20,
    "jac_thres": 0.2,
    "zero_thres": 1e-8,
    "max_iters": 200,
    "use_smooth": True,
    "scs_fallback": False,
    "normalize": True,
    "post_scal": True,
}
param_first_merge = {"thres_corr": 0.8}
param_second_spatial = {
    "dl_wnd": 5,
    "sparse_penal": 0.01,
    "update_background": True,
    "normalize": True,
    "size_thres": (25, None),
}
param_second_temporal = {
    "noise_freq": 0.06,
    "sparse_penal": 1,
    "p": 1,
    "add_lag": 20,
    "jac_thres": 0.2,
    "zero_thres": 1e-8,
    "max_iters": 500,
    "use_smooth": True,
    "scs_fallback": False,
    "normalize": True,
    "post_scal": True,
}

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = intpath

sys.path.append(minian_path)
from minian.cnmf import (
    compute_AtC,
    compute_trace,
    get_noise_fft,
    smooth_sig,
    unit_merge,
    update_spatial,
    update_temporal,
)
from minian.initialization import (
    gmm_refine,
    initA,
    initbf,
    initC,
    intensity_refine,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
)
from minian.motion_correction import apply_shifts, estimate_shifts
from minian.preprocessing import denoise, remove_background
from minian.utilities import load_videos, open_minian, save_minian
from minian.visualization import (
    CNMFViewer,
    VArrayViewer,
    generate_videos,
    visualize_gmm_fit,
    visualize_preprocess,
    visualize_seeds,
    visualize_spatial_update,
    visualize_temporal_update,
    write_video,
)

dpath = os.path.abspath(dpath)
hv.notebook_extension("bokeh", width=100)

cluster = LocalCluster(n_workers=1, memory_limit="8GB")
client = Client(cluster)

varr = load_videos(dpath, **param_load_videos)


def test_pre_check():
    dirpath = os.path.join(dpath, "minian")
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    minianvideo = os.path.join(dpath, "minian_mc.mp4")
    if os.path.exists(minianvideo):
        os.remove(minianvideo)
    assert os.path.exists(minianvideo) == False


def test_pipeline():
    global varr, subset_mc
    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(varr, framerate=5, summary=None)
        display(vaviewer.show())

    varr = save_minian(
        varr.rename("varr"),
        intpath,
        overwrite=True,
        chunks={"frame": 20, "height": -1, "width": -1},
    )

    if interactive:
        try:
            subset_mc = list(vaviewer.mask.values())[0]
        except IndexError:
            pass

    varr_ref = varr.sel(subset)

    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min

    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(
            [varr.rename("original"), varr_ref.rename("glow_removed")],
            framerate=5,
            summary=None,
            layout=True,
        )
        display(vaviewer.show())

    hv.output(size=output_size)
    if interactive:
        display(
            visualize_preprocess(
                varr_ref.isel(frame=0).compute(),
                denoise,
                method=["median"],
                ksize=[5, 7, 9],
            )
        )

    varr_ref = denoise(varr_ref, **param_denoise)

    hv.output(size=output_size)
    if interactive:
        display(
            visualize_preprocess(
                varr_ref.isel(frame=0),
                remove_background,
                method=["tophat"],
                wnd=[10, 15, 20],
            )
        )

    varr_ref = remove_background(varr_ref, **param_background_removal)

    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

    # Motion correction
    # def test_motion_correction():
    #     varr_ref = open_minian(intpath, "varr")
    shifts = estimate_shifts(varr_ref.sel(subset_mc), npart=34, **param_estimate_shift)

    shifts = save_minian(
        shifts.rename("shifts").chunk({"frame": 20}), **param_save_minian
    )

    test_shifts = open_minian(os.path.join(dpath_fixture, "minian"), "shifts")
    assert shifts.all() == test_shifts.all(), "Test Fail: arrays are not the same"

    hv.output(size=output_size)
    if interactive:
        display(
            hv.NdOverlay(
                dict(
                    width=hv.Curve(shifts.sel(variable="width")),
                    height=hv.Curve(shifts.sel(variable="height")),
                )
            )
        )

    Y = apply_shifts(varr_ref, shifts)
    Y = Y.fillna(0)

    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": 10, "width": 16},
    )

    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(
            [varr_ref.rename("before_mc"), Y_fm_chk.rename("after_mc")],
            framerate=5,
            summary=None,
            layout=True,
        )
        display(vaviewer.show())

    im_opts = dict(frame_width=500, aspect=752 / 480, cmap="Viridis", colorbar=True)
    (
        hv.Image(
            varr_ref.max("frame").compute(), ["width", "height"], label="before_mc"
        ).opts(**im_opts)
        + hv.Image(
            Y_hw_chk.max("frame").compute(), ["width", "height"], label="after_mc"
        ).opts(**im_opts)
    )

    vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
    write_video(vid_arr, "minian_mc.mp4", dpath)

    # Retrieve minian_mc.mp4 file inside fixture folder, used for testing
    fixture_probe = ffmpeg.probe(os.path.join(dpath_fixture, "minian_mc.mp4"))
    fixture_video_streams = [
        stream for stream in fixture_probe["streams"] if stream["codec_type"] == "video"
    ]

    # Check 'minian_mc.mp4' was written to folder, with same size as the one in fixture folder
    assert (
        os.path.exists(os.path.join(dpath, "minian_mc.mp4")) == True
    ), "minian_mc.mp4 was written to local folder"
    # Check if the sizes of the minian_mc.mp4 are +/- equal
    assert (
        abs(
            os.path.getsize(os.path.join(dpath, "minian_mc.mp4"))
            - os.path.getsize(os.path.join(dpath_fixture, "minian_mc.mp4"))
        )
        < 5000
    )

    probe = ffmpeg.probe(os.path.join(dpath, "minian_mc.mp4"))
    video_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    ]

    # Compare newly created minian_mc.mp4 file to the one in fixture folder
    assert video_streams[0]["width"] == fixture_video_streams[0]["width"]
    assert video_streams[0]["height"] == fixture_video_streams[0]["height"]
    assert video_streams[0]["codec_type"] == fixture_video_streams[0]["codec_type"]
    assert video_streams[0]["duration_ts"] == fixture_video_streams[0]["duration_ts"]
    assert (
        video_streams[0]["codec_long_name"]
        == fixture_video_streams[0]["codec_long_name"]
    )

    # Initialization
    # def test_initialization():
    #     Y_hw_chk = open_minian(intpath, "Y_hw_chk")
    #     Y_fm_chk = open_minian(intpath, "Y_fm_chk")
    max_proj = save_minian(
        Y_hw_chk.max("frame").rename("max_proj"), **param_save_minian
    ).compute()

    test_max_proj = open_minian(os.path.join(dpath_fixture, "minian"), "max_proj")
    assert max_proj.all() == test_max_proj.all(), "Test Fail: arrays are not the same"

    seeds = seeds_init(Y_fm_chk, **param_seeds_init)

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds)

    if interactive:
        noise_freq_list = [0.005, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
        example_seeds = seeds.sample(6, axis="rows")
        example_trace = (
            Y_hw_chk.stack(spatial=["height", "width"])
            .sel(
                spatial=[tuple(hw) for hw in example_seeds[["height", "width"]].values]
            )
            .assign_coords(spatial=np.arange(6))
            .rename(dict(spatial="seed"))
        )
        smooth_dict = dict()
        for freq in noise_freq_list:
            trace_smth_low = smooth_sig(example_trace, freq)
            trace_smth_high = smooth_sig(example_trace, freq, btype="high")
            trace_smth_low = trace_smth_low.compute()
            trace_smth_high = trace_smth_high.compute()
            hv_trace = hv.HoloMap(
                {
                    "signal": (
                        hv.Dataset(trace_smth_low)
                        .to(hv.Curve, kdims=["frame"])
                        .opts(frame_width=300, aspect=2, ylabel="Signal (A.U.)")
                    ),
                    "noise": (
                        hv.Dataset(trace_smth_high)
                        .to(hv.Curve, kdims=["frame"])
                        .opts(frame_width=300, aspect=2, ylabel="Signal (A.U.)")
                    ),
                },
                kdims="trace",
            ).collate()
            smooth_dict[freq] = hv_trace

    hv.output(size=output_size)
    if interactive:
        hv_res = (
            hv.HoloMap(smooth_dict, kdims=["noise_freq"])
            .collate()
            .opts(aspect=2)
            .overlay("trace")
            .layout("seed")
            .cols(3)
        )
        display(hv_res)

    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)

    if gmm:
        display(visualize_gmm_fit(pnr, gmm, 100))

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds, "mask_pnr")

    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds, "mask_ks")

    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds_final, "mask_mrg")

    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )

    A, C = unit_merge(A_init, C_init, **param_first_merge)

    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": 20}
    )

    b, f = initbf(Y_fm_chk, A, C_chk)
    b = save_minian(b.rename("b"), intpath, overwrite=True)
    f = save_minian(f.rename("f"), intpath, overwrite=True)

    im_opts = dict(
        frame_width=500,
        aspect=A.sizes["width"] / A.sizes["height"],
        cmap="Viridis",
        colorbar=True,
    )
    cr_opts = dict(frame_width=750, aspect=1.5 * A.sizes["width"] / A.sizes["height"])
    (
        hv.Image(
            A.sum("unit_id").rename("A").compute(), kdims=["width", "height"]
        ).opts(**im_opts)
        + hv.Image(C.rename("C").compute(), kdims=["frame", "unit_id"]).opts(
            cmap="viridis", colorbar=True, **cr_opts
        )
        + hv.Image(b.rename("b").compute(), kdims=["width", "height"]).opts(**im_opts)
        + datashade(
            hv.Curve(f.rename("f").compute(), kdims=["frame"]), min_alpha=200
        ).opts(**cr_opts)
    ).cols(2)

    # CNMF
    # def test_cnmf():
    try:
        client.close()
        cluster.close()
    except NameError:
        pass
    cluster = LocalCluster(
        n_workers=4, threads_per_worker=1, memory_limit="2GB", resources={"task": 1}
    )
    client = Client(cluster)

    # Y_hw_chk = open_minian(intpath, "Y_hw_chk")
    # A = open_minian(intpath, "A")
    # b = open_minian(intpath, "b")
    # C = open_minian(intpath, "C")
    # f = open_minian(intpath, "f")

    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C.sel(unit_id=units).persist()

    if interactive:
        sprs_ls = [0.005, 0.01, 0.05]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_b, cur_f, cur_mask = update_spatial(
                Y_hw_chk,
                A_sub,
                b,
                C_sub,
                f,
                sn_spatial,
                in_memory=True,
                dl_wnd=param_first_spatial["dl_wnd"],
                sparse_penal=cur_sprs,
            )
            if cur_A.sizes["unit_id"]:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    A_new, b_new, f_new, mask = update_spatial(
        Y_hw_chk, A, b, C, f, sn_spatial, **param_first_spatial
    )

    hv.output(size=output_size)
    opts = dict(
        plot=dict(height=A.sizes["height"], width=A.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    (
        hv.Image(A.sum("unit_id").compute().rename("A"), kdims=["width", "height"])
        .opts(**opts)
        .relabel("Spatial Footprints Initial")
        + hv.Image(
            (A.fillna(0) > 0).sum("unit_id").compute().rename("A"),
            kdims=["width", "height"],
        )
        .opts(**opts)
        .relabel("Binary Spatial Footprints Initial")
        + hv.Image(
            A_new.sum("unit_id").compute().rename("A"), kdims=["width", "height"]
        )
        .opts(**opts)
        .relabel("Spatial Footprints First Update")
        + hv.Image(
            (A_new > 0).sum("unit_id").compute().rename("A"),
            kdims=["width", "height"],
        )
        .opts(**opts)
        .relabel("Binary Spatial Footprints First Update")
    ).cols(2)

    hv.output(size=output_size)
    opts_im = dict(
        plot=dict(height=b.sizes["height"], width=b.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    opts_cr = dict(plot=dict(height=b.sizes["height"], width=b.sizes["height"] * 2))
    (
        hv.Image(b.compute(), kdims=["width", "height"])
        .opts(**opts_im)
        .relabel("Background Spatial Initial")
        + hv.Curve(f.compute(), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal Initial")
        + hv.Image(b_new.compute(), kdims=["width", "height"])
        .opts(**opts_im)
        .relabel("Background Spatial First Update")
        + hv.Curve(f_new.compute(), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal First Update")
    ).cols(2)

    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": 20}).rename("f"), intpath, overwrite=True)
    C = C.sel(unit_id=A.coords["unit_id"].values)
    C_chk = C_chk.sel(unit_id=A.coords["unit_id"].values)

    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C_chk.sel(unit_id=units).persist()

    if interactive:
        p_ls = [1]
        sprs_ls = [0.1, 0.5, 1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = (
            compute_trace(Y_fm_chk, A_sub, b, C_sub, f)
            .persist()
            .chunk({"unit_id": 1, "frame": -1})
        )
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(
            p_ls, sprs_ls, add_ls, noise_ls
        ):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print(
                "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(
                    cur_p, cur_sprs, cur_add, cur_noise
                )
            )
            cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                A_sub,
                C_sub,
                YrA=YrA,
                sparse_penal=cur_sprs,
                p=cur_p,
                use_smooth=True,
                add_lag=cur_add,
                noise_freq=cur_noise,
            )
            (
                YA_dict[ks],
                C_dict[ks],
                S_dict[ks],
                g_dict[ks],
                sig_dict[ks],
                A_dict[ks],
            ) = (
                YrA.compute(),
                cur_C.compute(),
                cur_S.compute(),
                cur_g.compute(),
                (cur_C + cur_b0 + cur_c0).compute(),
                A_sub.compute(),
            )
        hv_res = visualize_temporal_update(
            YA_dict,
            C_dict,
            S_dict,
            g_dict,
            sig_dict,
            A_dict,
            kdims=["p", "sparse penalty", "additional lag", "noise frequency"],
        )

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True
    )

    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param_first_temporal
    )

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
    (
        hv.Image(C.compute().rename("ci"), kdims=["frame", "unit_id"])
        .opts(**opts_im)
        .relabel("Temporal Trace Initial")
        + hv.Div("")
        + hv.Image(C_new.compute().rename("c1"), kdims=["frame", "unit_id"])
        .opts(**opts_im)
        .relabel("Temporal Trace First Update")
        + hv.Image(S_new.compute().rename("s1"), kdims=["frame", "unit_id"])
        .opts(**opts_im)
        .relabel("Spikes First Update")
    ).cols(2)

    hv.output(size=output_size)
    if interactive:
        h, w = A.sizes["height"], A.sizes["width"]
        im_opts = dict(aspect=w / h, frame_width=500, cmap="Viridis")
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = mask.where(mask == False, drop=True).coords["unit_id"].values
        if len(bad_units) > 0:
            hv_res = (
                hv.NdLayout(
                    {
                        "Spatial Footprint": Dynamic(
                            hv.Dataset(A.sel(unit_id=bad_units).compute().rename("A"))
                            .to(hv.Image, kdims=["width", "height"])
                            .opts(**im_opts)
                        ),
                        "Spatial Footprints of Accepted Units": Dynamic(
                            hv.Image(
                                A.sel(unit_id=mask)
                                .sum("unit_id")
                                .compute()
                                .rename("A"),
                                kdims=["width", "height"],
                            ).opts(**im_opts)
                        ),
                    }
                )
                + datashade(
                    hv.Dataset(YrA.sel(unit_id=bad_units).rename("raw")).to(
                        hv.Curve, kdims=["frame"]
                    )
                )
                .opts(**cr_opts)
                .relabel("Temporal Trace")
            ).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

    hv.output(size=output_size)
    if interactive:
        sig = C_new + b0_new + c0_new
        display(
            visualize_temporal_update(
                YrA.sel(unit_id=mask),
                C_new,
                S_new,
                g,
                sig,
                A.sel(unit_id=mask),
            )
        )

    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": 20}
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
    (
        hv.Image(C.compute().rename("c1"), kdims=["frame", "unit_id"])
        .relabel("Temporal Signals Before Merge")
        .opts(**opts_im)
        + hv.Image(C_mrg.compute().rename("c2"), kdims=["frame", "unit_id"])
        .relabel("Temporal Signals After Merge")
        .opts(**opts_im)
    )

    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": 20},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = sig.sel(unit_id=units).persist()

    if interactive:
        sprs_ls = [5e-3, 1e-2, 5e-2]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_b, cur_f, cur_mask = update_spatial(
                Y_hw_chk,
                A_sub,
                b,
                C_sub,
                f,
                sn_spatial,
                in_memory=True,
                dl_wnd=param_second_spatial["dl_wnd"],
                sparse_penal=cur_sprs,
            )
            if cur_A.sizes["unit_id"]:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    A_new, b_new, f_new, mask = update_spatial(
        Y_hw_chk, A, b, sig, f, sn_spatial, **param_second_spatial
    )

    hv.output(size=output_size)
    opts = dict(
        plot=dict(height=A.sizes["height"], width=A.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    (
        hv.Image(A.sum("unit_id").compute().rename("A"), kdims=["width", "height"])
        .opts(**opts)
        .relabel("Spatial Footprints Last")
        + hv.Image(
            (A.fillna(0) > 0).sum("unit_id").compute().rename("A"),
            kdims=["width", "height"],
        )
        .opts(**opts)
        .relabel("Binary Spatial Footprints Last")
        + hv.Image(
            A_new.sum("unit_id").compute().rename("A"), kdims=["width", "height"]
        )
        .opts(**opts)
        .relabel("Spatial Footprints New")
        + hv.Image(
            (A_new > 0).sum("unit_id").compute().rename("A"),
            kdims=["width", "height"],
        )
        .opts(**opts)
        .relabel("Binary Spatial Footprints New")
    ).cols(2)

    hv.output(size=output_size)
    opts_im = dict(
        plot=dict(height=b.sizes["height"], width=b.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    opts_cr = dict(plot=dict(height=b.sizes["height"], width=b.sizes["height"] * 2))
    (
        hv.Image(b.compute(), kdims=["width", "height"])
        .opts(**opts_im)
        .relabel("Background Spatial Last")
        + hv.Curve(f.compute(), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal Last")
        + hv.Image(b_new.compute(), kdims=["width", "height"])
        .opts(**opts_im)
        .relabel("Background Spatial New")
        + hv.Curve(f_new.compute(), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal New")
    ).cols(2)

    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": 20}).rename("f"), intpath, overwrite=True)
    C = C.sel(unit_id=A.coords["unit_id"].values)
    C_chk = C_chk.sel(unit_id=A.coords["unit_id"].values)

    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C_chk.sel(unit_id=units).persist()

    if interactive:
        p_ls = [1]
        sprs_ls = [0.1, 0.5, 1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = (
            compute_trace(Y_fm_chk, A_sub, b, C_sub, f)
            .persist()
            .chunk({"unit_id": 1, "frame": -1})
        )
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(
            p_ls, sprs_ls, add_ls, noise_ls
        ):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print(
                "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(
                    cur_p, cur_sprs, cur_add, cur_noise
                )
            )
            cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                A_sub,
                C_sub,
                YrA=YrA,
                sparse_penal=cur_sprs,
                p=cur_p,
                use_smooth=True,
                add_lag=cur_add,
                noise_freq=cur_noise,
            )
            (
                YA_dict[ks],
                C_dict[ks],
                S_dict[ks],
                g_dict[ks],
                sig_dict[ks],
                A_dict[ks],
            ) = (
                YrA.compute(),
                cur_C.compute(),
                cur_S.compute(),
                cur_g.compute(),
                (cur_C + cur_b0 + cur_c0).compute(),
                A_sub.compute(),
            )
        hv_res = visualize_temporal_update(
            YA_dict,
            C_dict,
            S_dict,
            g_dict,
            sig_dict,
            A_dict,
            kdims=["p", "sparse penalty", "additional lag", "noise frequency"],
        )

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True
    )

    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param_first_temporal
    )

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
    (
        hv.Image(C.compute().rename("c1"), kdims=["frame", "unit_id"])
        .opts(**opts_im)
        .relabel("Temporal Trace Last")
        + hv.Image(S.compute().rename("s1"), kdims=["frame", "unit_id"])
        .opts(**opts_im)
        .relabel("Spikes Last")
        + hv.Image(C_new.compute().rename("c2"), kdims=["frame", "unit_id"])
        .opts(**opts_im)
        .relabel("Temporal Trace New")
        + hv.Image(S_new.compute().rename("s2"), kdims=["frame", "unit_id"])
        .opts(**opts_im)
        .relabel("Spikes New")
    ).cols(2)

    hv.output(size=output_size)
    if interactive:
        h, w = A.sizes["height"], A.sizes["width"]
        im_opts = dict(aspect=w / h, frame_width=500, cmap="Viridis")
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = mask.where(mask == False, drop=True).coords["unit_id"].values
        if len(bad_units) > 0:
            hv_res = (
                hv.NdLayout(
                    {
                        "Spatial Footprint": Dynamic(
                            hv.Dataset(A.sel(unit_id=bad_units).compute().rename("A"))
                            .to(hv.Image, kdims=["width", "height"])
                            .opts(**im_opts)
                        ),
                        "Spatial Footprints of Accepted Units": Dynamic(
                            hv.Image(
                                A.sel(unit_id=mask)
                                .sum("unit_id")
                                .compute()
                                .rename("A"),
                                kdims=["width", "height"],
                            ).opts(**im_opts)
                        ),
                    }
                )
                + datashade(
                    hv.Dataset(YrA.sel(unit_id=bad_units).rename("raw")).to(
                        hv.Curve, kdims=["frame"]
                    )
                )
                .opts(**cr_opts)
                .relabel("Temporal Trace")
            ).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

    hv.output(size=output_size)
    if interactive:
        sig = C_new + b0_new + c0_new
        display(
            visualize_temporal_update(
                YrA.sel(unit_id=mask),
                C_new,
                S_new,
                g,
                sig,
                A.sel(unit_id=mask),
            )
        )

    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": 20}
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    try:
        client.close()
        cluster.close()
    except NameError:
        pass
    cluster = LocalCluster(n_workers=1, memory_limit="8GB")
    client = Client(cluster)

    AC = save_minian(compute_AtC(A, C_chk).rename("AC"), intpath, overwrite=True)
    generate_videos(varr, Y_fm_chk, AC=AC, vpath=dpath)

    if interactive:
        cnmfviewer = CNMFViewer(A=A, C=C, S=S, org=Y_fm_chk)

    hv.output(size=output_size)
    if interactive:
        display(cnmfviewer.show())

    if interactive:
        A = A.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        C = C.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        S = S.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        c0 = c0.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        b0 = b0.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))

    test_A = open_minian(os.path.join(dpath_fixture, "minian"), "A")
    assert (
        A.all() == test_A.all()
    ), "Test Fail: A does not match results in fixture folder"

    test_C = open_minian(os.path.join(dpath_fixture, "minian"), "C")
    assert (
        C.all() == test_C.all()
    ), "Test Fail: C does not match results in fixture folder"

    test_S = open_minian(os.path.join(dpath_fixture, "minian"), "S")
    assert (
        S.all() == test_S.all()
    ), "Test Fail: S does not match results in fixture folder"

    test_c0 = open_minian(os.path.join(dpath_fixture, "minian"), "c0")
    assert (
        c0.all() == test_c0.all()
    ), "Test Fail: c0 does not match results in fixture folder"

    test_b0 = open_minian(os.path.join(dpath_fixture, "minian"), "b0")
    assert (
        b0.all() == test_b0.all()
    ), "Test Fail: b0 does not match results in fixture folder"

    test_b = open_minian(os.path.join(dpath_fixture, "minian"), "b")
    assert (
        b.all() == test_b.all()
    ), "Test Fail: b does not match results in fixture folder"

    test_f = open_minian(os.path.join(dpath_fixture, "minian"), "f")
    assert (
        f.all() == test_f.all()
    ), "Test Fail: f does not match results in fixture folder"

    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)
