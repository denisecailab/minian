import pytest
import numpy as np
import holoviews as hv

from ..utilities import load_videos
from ..preprocessing import denoise, remove_background, stripe_correction

dpath = "./demo_movies"

param_load_videos = {
    "pattern": "msCam[0-9].avi",
    "dtype": np.uint8,
    "downsample": dict(frame=2, height=1, width=1),
    "downsample_strategy": "subset",
}

param_denoise = {"method": "median", "ksize": 7}

param_background_removal = {
    'method': 'tophat',
    'wnd': 15
}

@pytest.fixture
def varr():
    return load_videos(dpath, **param_load_videos)


def test_can_load_videos(varr):
    print("varr")
    print(varr)
    assert varr.shape[0] == 100  # frames
    assert varr.shape[1] == 480  # height
    assert varr.shape[2] == 752  # width
    return varr


def test_can_init_holoviews():
    hv.notebook_extension("bokeh")


def test_subset_part_video(varr):
    subset = dict(frame=slice(0, None))
    varr_ref = varr.sel(subset)
    assert varr_ref.all() == varr.all()

    
def test_remove_background(varr):
    varr_ref = denoise(varr, **param_denoise)
    varr_ref_remove = remove_background(varr_ref, **param_background_removal)
    assert (
        varr_ref.all() != varr_ref_remove.all()
    )  # when both are equal the denoise didn't do anything --> fail

def test_denoise(varr):
    varr_ref = denoise(varr, **param_denoise)
    assert (
        varr_ref.all() != varr.all()
    )  # when both are equal the denoise didn't do anything --> fail