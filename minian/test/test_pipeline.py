import numpy as np
import nbformat
import os
from nbconvert.preprocessors import ExecutePreprocessor
from ..utilities import open_minian


def test_pipeline_notebook():
    # run the notebook
    with open("./pipeline.ipynb") as nbf:
        nb = nbformat.read(nbf, as_version=nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=-1)
    ep.preprocess(nb)
    # load results
    minian_ds = open_minian("./demo_movies/minian")
    assert minian_ds.sizes["frame"] == 2000
    assert minian_ds.sizes["height"] == 480
    assert minian_ds.sizes["width"] == 752
    assert minian_ds.sizes["unit_id"] == 390
    assert (minian_ds["shifts"].sum("frame").values == np.array([-957, 2512])).all()
    assert int(minian_ds["max_proj"].sum().compute()) == 1645143
    assert int(minian_ds["C"].sum().compute()) == 128420990
    assert int(minian_ds["S"].sum().compute()) == 1424842
    assert int(minian_ds["A"].sum().compute()) == 390
    assert os.path.exists("./demo_movies/minian_mc.mp4")
    assert os.path.exists("./demo_movies/minian.mp4")
