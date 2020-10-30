import pytest
import numpy as np
import holoviews as hv

from ..utilities import load_videos
from ..motion_correction import *

dpath = "./minian/test/test_movie"

param_load_videos = {
    "pattern": "msCam[0-9].avi",
    "dtype": np.uint8,
    "downsample": dict(frame=2, height=1, width=1),
    "downsample_strategy": "subset",
}

param_estimate_shift = {
    'dim': 'frame',
    'max_sh': 20}

@pytest.fixture
def varr():
    return load_videos(dpath, **param_load_videos)


def test_estimate_shifts(varr):
    shifts = estimate_shifts(varr, **param_estimate_shift)
    assert (
        shifts.any() != varr.any()
    )
    
def test_apply_shifts(varr):
    shifts = estimate_shifts(varr, **param_estimate_shift)
    varr_ref = apply_shifts(varr, shifts)
    assert (
        varr_ref.any() != shifts.any()
    ) #If not equal, shifts have changed