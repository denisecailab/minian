import os
import subprocess

import numpy as np

from ..utilities import open_minian


def test_pipeline_notebook():
    os.makedirs("artifact")
    args = [
        "jupyter",
        "nbconvert",
        "--ExecutePreprocessor.timeout=-1",
        "--to",
        "notebook",
        "--output",
        "artifact/pipeline.ipynb",
        "--execute",
        "pipeline.ipynb",
    ]
    subprocess.run(args, check=True)
    minian_ds = open_minian("./demo_movies/minian")
    assert minian_ds.sizes["frame"] == 2000
    assert minian_ds.sizes["height"] == 480
    assert minian_ds.sizes["width"] == 752
    assert minian_ds.sizes["unit_id"] == 365
    assert (
        minian_ds["motion"].sum("frame").values.astype(int) == np.array([423, -239])
    ).all()
    assert int(minian_ds["max_proj"].sum().compute()) == 1501505
    assert int(minian_ds["C"].sum().compute()) == 137240727
    assert int(minian_ds["S"].sum().compute()) == 1245256
    assert int(minian_ds["A"].sum().compute()) == 365
    assert os.path.exists("./demo_movies/minian_mc.mp4")
    # assert os.path.exists("./demo_movies/minian.mp4")
