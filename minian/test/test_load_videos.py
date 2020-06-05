import numpy as np
import holoviews as hv

from ..utilities import load_videos

param_load_videos = {
    'pattern': 'msCam[0-9]+\.avi$',
    'dtype': np.uint8,
    'downsample': dict(frame=2,height=1,width=1),
    'downsample_strategy': 'subset'
}

dpath = "./demo_movies"

def test_can_load_videos():
    varr = load_videos(dpath, **param_load_videos)
    assert varr.shape[0] == 1000  # frames
    assert varr.shape[1] == 480  # height
    assert varr.shape[2] == 752  # width
    return varr

def test_can_init_holoviews():
    hv.notebook_extension('bokeh')
