Cross-registration
==================

Cross-registration allows the user to register the outcomes of the
MiniAn pipeline across multiple experimental sessions. This script is
not necessary to analyze the data, but it is a useful add-on to deal
with longitudinal experiments.

Specify Directory and File Information
======================================

For cross-registration to work, the user needs to place the videos that
need to be cross-registered and their corresponding minian output data
in the same folder, following the scheme decribed in the Read The Docs.
In brief if data for subject 1 was acquired over three days, with two
sessions a day, the folders structure would be: Subject1 (parent folder)
with three subfolders for the days (2020_01_01, 2020_01_02, 2020_01_03),
with every subfolder having session 1 and session 2 as subfolders.
Inside every session folder there would be the videos to be
cross-registered and their corresponding minian output data.
``minian_path`` is the directory containing minian, ``dpath`` is the
path to the parent folder, ``f_pattern`` is a regular expression
identifying the naming pattern of minian output folders with a regex
expression (e.g. ``'minian$'``, or ``r'minian\.[0-9]+$'`` if data is
batch processed and has a timestamp), and ``id_dims`` should be a list
containing metadata identifiers used when analyzing the individual
sessions (e.g. ``['session','animal']``).

.. code:: ipython3

    minian_path = "."
    dpath = "./demo_movies"
    f_pattern = r'minian\.[0-9]+$' 
    id_dims = ['animal','session']

Define Paramaters
=================

``param_t_dist`` defines the maximal distance between cell centroids (in
pixel units) on different sessions to consider them as the same cell

.. code:: ipython3

    param_t_dist = 5
    output_size = 90

Load Modules
============

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2
    import os
    import sys
    import warnings
    sys.path.append(minian_path)
    import itertools as itt
    import numpy as np
    import xarray as xr
    import holoviews as hv
    import pandas as pd
    from holoviews.operation.datashader import datashade, regrid
    from dask.diagnostics import ProgressBar
    from minian.cross_registration import (calculate_centroids, calculate_centroid_distance, calculate_mapping,
                                           group_by_session, resolve_mapping, fill_mapping)
    from minian.motion_correction import estimate_shifts, apply_shifts
    from minian.utilities import open_minian, open_minian_mf
    from minian.visualization import AlignViewer
    hv.notebook_extension('bokeh', width=100)
    pbar = ProgressBar(minimum=2)
    pbar.register()

Allign Videos
=============

open datasets
-------------

.. code:: ipython3

    minian_ds = open_minian_mf(
        dpath, id_dims, pattern=f_pattern)

estimate shifts
---------------

.. code:: ipython3

    %%time
    temps = minian_ds['max_proj'].compute().rename('temps')
    shifts = estimate_shifts(temps, max_sh=20, dim='session').compute().rename('shifts')
    temps_sh = apply_shifts(temps, shifts).compute().rename('temps_shifted')
    shiftds = xr.merge([temps, shifts, temps_sh])

visualize alignment
-------------------

.. code:: ipython3

    hv.output(size=output_size)
    opts_im = {
        'aspect': shiftds.sizes['width'] / shiftds.sizes['height'],
        'frame_width': 500, 'cmap': 'viridis'}
    hv_temps = (hv.Dataset(temps).to(hv.Image, kdims=['width', 'height'])
                .opts(**opts_im).layout('session').cols(1))
    hv_temps_sh = (hv.Dataset(temps_sh).to(hv.Image, kdims=['width', 'height'])
                .opts(**opts_im).layout('session').cols(1))
    display(hv_temps + hv_temps_sh)

visualize overlap of field of view across all sessions
------------------------------------------------------

.. code:: ipython3

    hv.output(size=output_size)
    opts_im = {
        'aspect': shiftds.sizes['width'] / shiftds.sizes['height'],
        'frame_width': 500, 'cmap': 'viridis'}
    window = shiftds['temps_shifted'].isnull().sum('session')
    window, temps_sh = xr.broadcast(window, shiftds['temps_shifted'])
    hv_wnd = hv.Dataset(window, kdims=list(window.dims)).to(hv.Image, ['width', 'height'])
    hv_temps = hv.Dataset(temps_sh, kdims=list(temps_sh.dims)).to(hv.Image, ['width', 'height'])
    hv_wnd.opts(**opts_im).relabel("Window") + hv_temps.opts(**opts_im).relabel("Shifted Templates")

apply shifts and set window
---------------------------

.. code:: ipython3

    A_shifted = apply_shifts(minian_ds['A'].chunk(dict(height=-1, width=-1)), shiftds['shifts'])

.. code:: ipython3

    def set_window(wnd):
        return wnd == wnd.min()
    window = xr.apply_ufunc(
        set_window,
        window,
        input_core_dims=[['height', 'width']],
        output_core_dims=[['height', 'width']],
        vectorize=True)

Calculate Centroid Distance
===========================

calculate centroids
-------------------

.. code:: ipython3

    %%time
    cents = calculate_centroids(A_shifted, window)

calculate centroid distance
---------------------------

.. code:: ipython3

    %%time
    dist = calculate_centroid_distance(cents, index_dim=['animal'])

Get Overlap Across Sessions
===========================

threshold overlap based upon centroid distance and generate overlap mappings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    dist_ft = dist[dist['variable', 'distance'] < param_t_dist].copy()
    dist_ft = group_by_session(dist_ft)

.. code:: ipython3

    %%time
    mappings = calculate_mapping(dist_ft)
    mappings_meta = resolve_mapping(mappings)
    mappings_meta_fill = fill_mapping(mappings_meta, cents)
    mappings_meta_fill.head()

save results
~~~~~~~~~~~~

.. code:: ipython3

    mappings_meta_fill.to_pickle(os.path.join(dpath, "mappings.pkl"))
    cents.to_pickle(os.path.join(dpath, "cents.pkl"))
    shiftds.to_netcdf(os.path.join(dpath, "shiftds.nc"))

View Overlap Across Any 3 Sessions
==================================

.. code:: ipython3

    mappings_meta_fill = pd.read_pickle(os.path.join(dpath, "mappings.pkl"))
    cents = pd.read_pickle(os.path.join(dpath, "cents.pkl"))
    shiftds = xr.open_dataset(os.path.join(dpath, 'shiftds.nc'))

.. code:: ipython3

    hv.output(size=output_size)
    alnviewer = AlignViewer(minian_ds, cents, mappings_meta_fill, shiftds)
    alnviewer.show()
