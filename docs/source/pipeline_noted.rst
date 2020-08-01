About this document
===================

The purpose of this annotated version of the minian pipeline is to guide
the user through each step of the code, working with a short demo movie.
The intention is to enable the user to understand the code as much as
possible so that they are equipped with the knowledge necessary to
customize the code for their own needs, regardless of prior programming
skills. This version is **not** supposed to be run as a production tool
for analyzing your own videos. If you click “Run All” on this document,
you **will** likely encounter errors. For the purposes of your own
analyses, consider modifying the accompanying **pipeline.ipynb** and
**batch_processing.ipynb** to suit your needs.

Before we start, it s highly recommended that you get familiar with
basic python concepts and operations like `string
manipulation <https://docs.python.org/3.4/library/string.html>`__,
`tuples, lists and
dictionaries <https://docs.python.org/3/tutorial/datastructures.html>`__,
as well as a little bit about `object-oriented
programming <https://python.swaroopch.com/oop.html>`__ and `python
modules <https://docs.python.org/3/tutorial/modules.html>`__.

Another note on the styling of this document: most of the sentences
should hopefully make sense if taken literally. However, some special
formatting of the text is used to demonstrate the close relationship
between the concepts discussed and the code, as well as encouraging the
reader to understand the Python syntax. Specifically:

-  a `hyperlink <https://en.wikipedia.org/wiki/Hyperlink>`__ usually
   points to a well-defined python module, class or methods, especially
   when that concept is first encountered in this document. The link
   usually points to the official documentation of that concept, which
   in some cases might not be the best place to start for a beginner. If
   you find the documentation puzzling, try to google the concept in
   question and find a tutorial that best suits you.

-  an inline ``code`` usually refers to a name that already exsists in
   the
   `namespace <https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces>`__
   (i.e. the context where we run the codes in this document). It can be
   a previously encountered concept, but more often it referes to
   variable names or method names that we
   `imported <https://docs.python.org/3/reference/import.html>`__ or
   have defined along the way.

-  **bold** texts are used more loosely to highlight anything that
   requires more attention. Though they are not used as carefully as
   previous formats, they often refer to specific values that a variable
   or method arguments can assume.

-  

   .. container:: alert alert-info

      Blue tip boxes are used to provide direct instructions and coding
      tips to help users run through this pipeline smoothly! :)

-  

   .. container:: alert alert-success

      Green tip boxes are used to let the user know what to expect from
      the output visualization result of parameter exploring.

Workflow
========

As shown in the workflow below, there are 7 main sections in this
pipeline. Results will be saved along the way, following **Motion
correction**, **Background removal**, **Initialization**, and **CNMF**.
Therefore, when you run through the pipeline, if you decide to
restart/shutdown the kernal before you finish, you don’t have to re-run
everything the next time you want to pick this up (with the exception of
the **Setting Up** module). For example, if you restart/shutdown the
kernal after **Initialization**, when you return to your data you can
simply rerun the **Setting Up** module, and then go directly to
**CNMF**.

Before we dive into the pipeline, we will also introduce the most
powerful and important aspect of this pipeline – parameter exploring.
**CNMF** has been daunting to some because of the many parameters
involved in its implementation. The purpose of minian’s interactive
visualization steps is to make the impact of all parameters transparent
to users by allowing them to view the direct impact of parameter
manipulation on the data, helping them to find the parameters best
suited to their data. We will go into greater detail on how to use and
interpret these viusalization steps later on, but for now, just know
that if you are daunted by CNMF, these steps will likely be the most
helpful.

.. figure:: img/Workflow_v2.PNG
   :alt: workflow

   workflow

Setting up
==========

The cells under this section should be executed every time the kernel is
restarted.

load modules
------------

The following cell loads the necessary modules for minian to run and
usually should not be modified.

.. code:: ipython3

    %%capture
    %load_ext autoreload
    %autoreload 2
    import sys
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    import gc
    import psutil
    import numpy as np
    import xarray as xr
    import holoviews as hv
    import matplotlib.pyplot as plt
    import bokeh.plotting as bpl
    import dask.array as da
    import pandas as pd
    import dask
    import datashader as ds
    import itertools as itt
    import papermill as pm
    import ast
    import functools as fct
    from holoviews.operation.datashader import datashade, regrid, dynspread
    from datashader.colors import Sets1to3
    from dask.diagnostics import ProgressBar
    from IPython.core.display import display, HTML

set path and parameters
-----------------------

Here we set all of the parameters that control how the notebook will
behave. Ideally, the following cell would be the only thing you have to
change when analyzing different datasets. Indeed, we put all parameters
here under a single cell to facilitate batch processing. Most of the
parameters will not make sense until we reach and discuss the
corresponding steps in the pipeline. For this reason, here we discuss
only some initial parameters that have a broader impact, and leave the
discussion of specific parameters for later.

-  ``minian_path`` should be the path that contains a folder named
   **“minian”** , under which the actual code of **minian** (.py files)
   reside. The default value **.** means “current folder”, which should
   work in most cases, unless you want to try out another version of
   minian that is not in the same folder as this notebook.

-  ``dpath`` is the folder that contains the actual videos to be
   processed, which are usually named **“msCam*.avi”** where \* is a
   number.

-  | ``interactive`` controls whether interactive plots will be shown.
     Note that interactive plotting requires computation to be carried
     out, and thus could be very inefficient when the data are not in
     the memory (in particular, those steps where video is played).
     However, if your video is large, you may want to set
     ``interactive = True`` for the purposes of paramater exploration,
     but have ``in_memory`` set to false.

   .. container:: alert alert-info

      In practice, when you want to visualize interactive plots while
      figuring out the best parameters for your data, you would want
      in_memory=True and interactive=True, as long as your data can fit
      in the memory. On the other hand, once you finalize your
      parameters and are ready for batch processing, you want both of
      them to be set as False.

-  ``output_size`` controls the relative size of all the plots on a
   scale of 0-100 percent, though it can be set to values >100 without
   any problem. Adjust this to please your eye.

-  ``param_save_minian`` specifies how data is to be saved. ``dpath`` is
   the folder path defining where you want the data to be saved. We
   recommand using the same dpath as where you load the imaging data
   from. ``fname`` is the name of your dataset. In ``backend``,
   ``'zarr'`` is designed for parallel and out-of-core computation, and
   is the current default for minian. That said, its support is
   experimental for now and does not support incremental writing
   (i.e. it will be a pain to update part of an exsisting dataset).
   ``meta_dict`` is a ``dictionary`` that is used to construct meta data
   for the final labeled data structure.

.. container:: alert alert-info

   The defult meta_dict is assumes data is stored in heirarchiically
   arranged folders, as shown below. We recommand users to structure
   their data like this so that they don’t have to adjust this meta_dict
   setting. This is also the default manner in which Miniscope data is
   saved. However, if you already have a preferred way to store your
   data, you can simply change the value of meta_dict in this parameter
   to suit your needs!

**recommended folder structure**

.. figure:: img/folder_structure.png
   :alt: Folder Structure

   Folder Structure

The default value can be read as follows: the name of the last folder
(``values``\ =-1) in ``dpath`` (the folder that directly contains the
videos) will be used to designate the value of a field named
**‘session_id’**. The name of the second-to-last folder
(``values``\ =-2) in ``dpath`` will be used to designate the value for
**‘session’** and so on. Both the ``keys`` (field names) and ``values``
(numbers indicating which level of folder name should be used) of
``meta_dict`` can be modified to suit your data structure. ``overwrite``
is a boolean value controlling whether the data is overwritten if a file
already exsists. We set it to ``True`` here so you can easily play with
the demo multiple times, but **use extreme caution** with this during
actual analysis – in addition to erasing prior data that may be
important to you, under certain circumstances it is possible for
existing file structures to cause compatibablity issues and data will be
saved improperly. If you want to re-analyze a video from scratch using
different parameters, it is recommended that you delete existing data
first.

.. code:: ipython3

    #Set up Initial Basic Parameters#
    minian_path = "."
    dpath = "./demo_movies"
    subset = dict(frame=slice(0,None))
    subset_mc = None
    interactive = True
    output_size = 100
    param_save_minian = {
        'dpath': dpath,
        'fname': 'minian',
        'backend': 'zarr',
        'meta_dict': dict(session_id=-1, session=-2, animal=-3),
        'overwrite': True}
    
    #Pre-processing Parameters#
    param_load_videos = {
        'pattern': 'msCam[0-9]+\.avi$',
        'dtype': np.uint8,
        'downsample': dict(frame=2,height=1,width=1),
        'downsample_strategy': 'subset'}
    param_denoise = {
        'method': 'median',
        'ksize': 7}
    param_background_removal = {
        'method': 'tophat',
        'wnd': 15}
    
    #Motion Correction Parameters#
    subset_mc = None
    param_estimate_shift = {
        'dim': 'frame',
        'max_sh': 20}
    
    #Initialization Parameters#
    param_seeds_init = {
        'wnd_size': 1000,
        'method': 'rolling',
        'stp_size': 500,
        'nchunk': 100,
        'max_wnd': 15,
        'diff_thres': 2}
    param_pnr_refine = {
        'noise_freq': 0.1,
        'thres': 1,
        'med_wnd': None}
    param_ks_refine = {
        'sig': 0.05}
    param_seeds_merge = {
        'thres_dist': 5,
        'thres_corr': 0.7,
        'noise_freq': 0.1}
    param_initialize = {
        'thres_corr': 0.8,
        'wnd': 15,
        'noise_freq': 0.1}
    
    #CNMF Parameters#
    param_get_noise = {
        'noise_range': (0.1, 0.5),
        'noise_method': 'logmexp'}
    param_first_spatial = {
        'dl_wnd': 15,
        'sparse_penal': 0.1,
        'update_background': True,
        'normalize': True,
        'zero_thres': 'eps'}
    param_first_temporal = {
        'noise_freq': 0.1,
        'sparse_penal': 0.05,
        'p': 1,
        'add_lag': 20,
        'use_spatial': False,
        'jac_thres': 0.2,
        'zero_thres': 1e-8,
        'max_iters': 200,
        'use_smooth': True,
        'scs_fallback': False,
        'post_scal': True}
    param_first_merge = {
        'thres_corr': 0.8}
    param_second_spatial = {
        'dl_wnd': 15,
        'sparse_penal': 0.005,
        'update_background': True,
        'normalize': True,
        'zero_thres': 'eps'}
    param_second_temporal = {
        'noise_freq': 0.1,
        'sparse_penal': 0.05,
        'p': 1,
        'add_lag': 20,
        'use_spatial': False,
        'jac_thres': 0.2,
        'zero_thres': 1e-8,
        'max_iters': 500,
        'use_smooth': True,
        'scs_fallback': False,
        'post_scal': True}

import minian
-------------

The following cell loads **minian** and usually should not be modified.
If you encounter an ``ImportError``, check that you followed the
installation instructions and that ``minian_path`` is pointing to the
right place.

.. code:: ipython3

    %%capture
    sys.path.append(minian_path)
    from minian.utilities import load_videos, open_minian, save_minian, get_optimal_chk, rechunk_like
    from minian.preprocessing import denoise, remove_background
    from minian.motion_correction import estimate_shifts, apply_shifts
    from minian.initialization import seeds_init, gmm_refine, pnr_refine, intensity_refine, ks_refine, seeds_merge, initialize
    from minian.cnmf import get_noise_fft, update_spatial, compute_trace, update_temporal, unit_merge, smooth_sig
    from minian.visualization import VArrayViewer, CNMFViewer, generate_videos, visualize_preprocess, visualize_seeds, visualize_gmm_fit, visualize_spatial_update, visualize_temporal_update, write_video

module initialization
---------------------

The following cell handles initialization of modules and parameters
necessary for minian to be run and usually should not be modified.

.. code:: ipython3

    dpath = os.path.abspath(dpath)
    hv.notebook_extension('bokeh')
    if interactive:
        pbar = ProgressBar(minimum=2)
        pbar.register()

Pre-processing
==============

In the pre-processing steps that follow, videos will be loaded and any
initial processing (downsampling, subsetting, denoising) will be
performed.

All functions are evaluated lazily, which means that initially only a
“plan” for the actual computation will be created, without its
execution. Actual computations are carried out only when results are
being saved.

loading videos and visualization
--------------------------------

Recall the values of ``param_load_videos``:

.. code:: python

   param_load_videos = {
       'pattern': 'msCam[0-9]+\.avi$',
       'dtype': np.uint8,
       'downsample': dict(frame=2),
       'downsample_strategy': 'subset'}

The first argument of ``load_videos`` should be the path that contains
the videos, which is the file folder we previously defined (``dpath``)
we already defined. We then pass the dictionary, ``param_load_videos``,
defined earlier, which specifies several relevant arguments. The
argument ``pattern`` is optional and is the `regular
expression <https://docs.python.org/3/library/re.html>`__ used to filter
files under the specified folder. The default value
**‘msCam[0-9]+.avi$’** means that a file can only be loaded if its
filename contains **‘msCam’**, followed by at least one number, then
**‘.avi’** as the end of the filename. This can be changed to suit the
naming convention of your videos. ``dtype`` is the underlying `data
type <https://docs.scipy.org/doc/numpy-1.15.0/user/basics.types.html>`__
of the data. Usually ``uint8`` is good and should be preferred to save
memory demand. The resulting “video array” ``varr`` contains three
dimensions: ``height``, ``width``, and ``frame``. If you wish to
downsample the video, pass in a ``dictionary`` to ``downsample``, whose
keys should be the name of dimensions and values an integer specifying
how many times that dimension should be reduced. For example,
``downsample=dict('frame'=2)`` will temporally downsample the video with
a factor of 2. Instead, if you do not wish to downsample your data,
simply pass in ``downsample=None``. ``downsample_strategy`` will assume
two values: either ``'subset'``, meaning downsampling are carried out
simply by subsetting the data, or ``'mean'``, meaning a mean will be
calculated on the window of downsampling (the latter being slower).

``param_load_videos['downsample']`` should be specified as a python
`dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`__,
whose ``keys`` are the dimensions along which subsetting should be done,
and whose ``values`` specify how subsetting should be done.

.. container:: alert alert-info

   This is a good opportunity to introduce how to manipulate parameters
   while you are running through this pipeline. If you want to modify
   the parameters, you can either go back to the initial parameter
   setting cell and change things there. Alternatively, you can add a
   cell of code in which you change one or more paramaters. For example,
   if you want to change the downsampling setting for your data, you
   can:

**Option 1–Go back to initial parameter setting code cell, change the
parameters setting there, then rerun the parameter setting cell:**

**Example 1: Stop downsampling**

.. code:: python

   param_load_videos = {
       'pattern': 'msCam[0-9]+\.avi$',
       'dtype': np.float32,
       'in_memory': in_memory,
       'downsample': dict(frame=2),
       'downsample_strategy': 'subset'}

**change this to:**

.. code:: python

   param_load_videos = {
       'pattern': 'msCam[0-9]+\.avi$',
       'dtype': np.float32,
       'in_memory': in_memory,
       'downsample': None,
       'downsample_strategy': 'subset'}

**Example 2: Changing the downsampling setting from by ``frame`` to by
``height`` and ``width``, and also changing the downsampling strategy to
``mean``.**

.. code:: python

   param_load_videos = {
       'pattern': 'msCam[0-9]+\.avi$',
       'dtype': np.float32,
       'in_memory': in_memory,
       'downsample': dict(frame=2),
       'downsample_strategy': 'subset'}

**change this to:**

.. code:: python

   param_load_videos = {
       'pattern': 'msCam[0-9]+\.avi$',
       'dtype': np.float32,
       'in_memory': in_memory,
       'downsample': dict(height=2,width=2),
       'downsample_strategy': 'mean'}

**Option 2–Insert a code cell by clicking the little + symbol on the top
row of jupyter notebook. Then change the specific ‘keys’ with the
‘value’ you want to asign to them as shown below, and run this new code
cell.**

**Example 1: Stop downsampling**

.. code:: python

   param_load_videos['downsample'] = None

**Example 2: Changing the downsampling setting from by ``frame`` to by
``height`` and ``width``, and also changing the downsampling strategy to
``mean``.**

.. code:: python

   param_load_videos['downsample'] = dict(height=2,width=2)
   param_load_videos['strategy'] = 'mean'

.. code:: ipython3

    %%time
    varr = load_videos(dpath, **param_load_videos)
    chk = get_optimal_chk(varr.astype(float), dim_grp=[('frame',), ('height', 'width')])

The previous code cell loaded the videos and concatenated them together
into the unitary data object ``varr``, which is a
`xarray.DataArray <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html#xarray.DataArray>`__.
Now is a perfect time to familiarize yourself with this data structure
and the `xarray <https://xarray.pydata.org/en/stable/>`__ module in
general, since we will be using these data structures throughout the
analysis. Basically, a ``xarray.DataArray`` is a labeled N-dimensional
array, with many useful properties that make them easy to manipulate. We
can ask the computer to print out some information of ``varr`` by
calling its name (as with any other variable):

.. code:: ipython3

    varr

visualize raw data and optionally set roi for motion correction
---------------------------------------------------------------

We can see now that ``varr`` is a ``xarray.DataArray`` with a
`name <https://xarray.pydata.org/en/stable/generated/xarray.DataArray.name.html#xarray.DataArray.name>`__
``'demo_movies'`` and three dimensions: ``frame``, ``height`` and
``width``; and each dimension is labeled with ascending natural numbers.
The
`dtype <https://xarray.pydata.org/en/stable/generated/xarray.DataArray.dtype.html#xarray.DataArray.dtype>`__
(`data
type <https://docs.scipy.org/doc/numpy-1.14.0/user/basics.types.html>`__)
of ``varr`` is ``numpy.uint8``

In addition to this information, we can visualize ``varr`` with the help
of ``VArrayViewer``, which shows the array as a movie. You can also plot
summary traces like mean fluorescnece across ``frame`` by passing in a
``list`` of names of traces you want. Currently ``"mean"``, ``"min"``,
``"max"`` and ``"diff"`` are supported, where ``"diff"`` is mean
fluorescent value difference across all pixels in a ``frame``.

Finally ``VArrayViewer`` support a box drawing tool where you can draw
an arbitrary box in the field of view and record this box as a mask
using the “save mask” button. The mask is saved as ``vaviewer.mask``.
This mask could be useful for other steps, for example, when you want to
run motion correction on a sub-region of field of view.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(varr, framerate=5, summary=['mean', 'max'])
        display(vaviewer.show())

if you decided to set a mask for motion correction, the following cell
is an example of how to convert the mask into a ``subset_mc`` parameter
that can be later passed into motion correction functions.

.. code:: ipython3

    if interactive:
        try:
            subset_mc = list(vaviewer.mask.values())[0]
        except IndexError:
            pass

subset part of video
--------------------

Before proceeding to pre-processing, it’s good practice to check if
there is anything obviously wrong with the video (e.g. the camera
suddenly dropped, resulting in dark frames). This can usually be
observed by visualizing the video and checking the mean fluorescence
plot. To take out bad ``frame``\ s, let’s say, ``frame`` after 800, we
can utilize the
`xarray.DataArray.sel <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.sel.html>`__
method and
`slice <https://docs.python.org/3/library/functions.html#slice>`__:

.. container:: alert alert-info

   Here is good chance to briefly introduce how to use .sel and slice
   properly since this will be super useful to handle your imaging video
   data. See below:

By using DataArray.sel, you can easily use the basic format:

.. code:: python

   subsetDataArray = DataArray.sel(dimsA=object(value))

This will assign the data array you selected out to the new data array
which is ``subsetDataArray`` in this case.

Using slice() function, you can select a subset or a portion of your
data by calling the function like this:

.. code:: python

   subsetDimsA = slice(start, stop, step)

You can combine xarray.DataArray.sel method and slice function to
manipulate your multidimensional imaging data! Note that slice object is
just one of the useful objects that you can use here.

**Example 1**: Say that you want to get rid of the frames after 800:

.. code:: python

   varr_ref = varr.sel(frame=slice(None, 800))

This will subset ``varr`` along the ``frame`` dimension from the
begining (``None``) to the ``frame`` labeled **800**, then assign the
result back to ``varr_ref``, which is equivalent to taking out ``frame``
from **801** to the end. Note you can do the same thing to other
dimensions like ``height`` and ``width`` to take out certain pixels of
your video for all ``frame``\ s. For more information on using
``xarray.DataArray.sel``, as well as other indexing strategies, see
`xarray
documentation <http://xarray.pydata.org/en/stable/indexing.html>`__

**Example 2**: If you want to get rid of the timestamp located in the
last row of pixels in height dimention for the wireless Miniscope
recording, **.isel** will be a useful. See `xarray.Dataset.isel
documentation <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.isel.html>`__.
Different from **.sel**, **.isel** subsets out data by index, rather
than by coordinate:

.. code:: python

   varr_ref = varr.isel(height=slice(None, -1))

This will subset ``varr`` along the ``height`` dimension from the
begining to the second-to-last row of pixels, then assign the result
back to ``varr_ref``, which is equivalent to taking out last row of
``height``.

If your ``varr`` is fine, just assign it to ``varr_ref`` to keep the
naming consitent with later code.

**In production mode** – pipeline.ipynb or batch_processing.ipynb – we
usually use the ``subset`` parameter defined above under the module
paramater selection to control subsetting. Recall the ``subset``
parameter.

.. code:: python

   subset = None

``subset`` is used to subset the data. ``subset`` should be specified as
python
`dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`__,
whose ``keys`` are the dimensions along which subsetting should be done,
and whose ``values`` should specify how subsetting should be done
(usually a `slice
object <https://docs.python.org/3/c-api/slice.html>`__). A good usecase
for this is to take out some troublesome frames or to take out some bad
pixels. Alternatively, perhaps you are only interested in analalyzing
the first few minutes of a much longer recording session.

.. container:: alert alert-info

   Here is good chance to introduce how to create a dictionary, and more
   importantly, how to use it! See below:

The basic format of a dictionary takes the following form:

.. code:: python

   dictionary = {'key1': value1, 'key2': value2, ... 'keyN': valueN}

For example, if you only want to keep the first 800 ``frames``, and the
``height`` and ``width`` from 100 to 200, you could create this subset
dictionary:

.. code:: python

   subset = {
       'frame': slice(0, 800),
       'height': slice(100, 200),
       'width': slice(100, 200)}

Similarly, after you have a dictionary, you can call a specific key in
this dictionary and change the value accordingly. We can use the same
example here, say if you want to get rid of ``frames`` from 801 to the
end:

.. code:: python

   subset['frame'] = slice(None, 800)

.. container:: alert alert-info

   Leaving subset in default setting will result in no selection and is
   thus equivalent to assigning varr back to varr_ref.

.. code:: ipython3

    varr_ref = varr.sel(subset)

glow removal and visualization
------------------------------

Here we remove the general glow background caused by viganetting effect.
We simply calculate a minimum projection across all ``frame``\ s and
subtract that projection from all ``frame``\ s. A benefit of doing this
is you could interpret the result as “change of fluorescence from
baseline”, while preserving the linear scale of the raw data, which is
usually the range of a 8-bit integer – 0-255. The result can be
visualized again with ``VArrayViewer``

.. code:: ipython3

    %%time
    varr_min = varr_ref.min('frame').compute()
    varr_ref = varr_ref - varr_min

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(
            [varr.rename('original'), varr_ref.rename('glow_removed')],
            framerate=5,
            summary=None,
            layout=True)
        display(vaviewer.show())

denoise
-------

This step carries out denoising of the video frame by frame, using the
``denoise`` function. The function ``denoise`` takes in two required
arguments: the first is the video array to be processed (``varr_reff``),
and the second, ``method``, is a string specifying the denoising method
to use. Right now three methods are supported: ``'gaussian'``,
``'median'`` and ``'anisotropic'``. Under the hood, ``denoise`` simply
calls another function frame by frame in a parallel fashion. For
``method='gaussian'`` it calls
`GaussianBlur <https://www.docs.opencv.org/3.3.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1>`__
from the ``OpenCV`` package. For\ ``method='median'`` it calls
`MedianBlur <https://docs.opencv.org/3.4.3/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9>`__
from the ``OpenCV`` package. For ``method='anisotropic'`` it calls
`anisotropic_diffusion <http://loli.github.io/medpy/generated/medpy.filter.smoothing.anisotropic_diffusion.html>`__
from the ``medpy`` package. All additional `keyword
arguments <https://docs.python.org/3.7/tutorial/controlflow.html#keyword-arguments>`__
passed into ``denoise`` are directly passed into one of those two
denoising functions under the hood.

Recall that by default we use a median filter for enoising:

.. code:: python

   param_first_denoise = {
       'method': 'median',
       'ksize': 5}

There is only one parameter controlling how the filtering is done: the
kernel size (``ksize``) of the filter. The effect of this parameter can
be visualized with the tool below.

.. container:: alert alert-info

   Generally ksize=5 is good (approximately half the diamater of the
   largest cell). Note that if you do want to play with the ksize, it
   has to be odd number.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(visualize_preprocess(varr_ref.isel(frame=0), denoise, method=['median'], ksize=[5, 7, 9]))

.. code:: ipython3

    varr_ref = denoise(varr_ref, **param_denoise)

backgroun removal
-----------------

Recall the parameters for background removal:

.. code:: python

   param_background_removal = {
       'method': 'tophat',
       'wnd': 10}

This step attempts to estimate background (everything except the
fluorescent signal of in-focus cells) frame by frame and remove it. As
with the last step, the first argument to ``remove_background`` is our
video (``varr_mc``), and the second is the ``method`` to use for
background subtraction. There are two methods available: ``'uniform'``
or ``'tophat'``. Both require a single parameter - a window size
(``wnd``), which is the third required argument to
``remove_background``. The two methods differ in how background is
estimated.

For ``method='tophat'``, a `disk
element <http://scikit-image.org/docs/dev/api/skimage.morphology.html#disk>`__
with a radius of ``wnd`` is created. Then, a `morphological
erosion <https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm>`__ using
the disk element is applied to each frame, which eats away any bright
“features” that are smaller than the disk element. Subsequently, a
`morphological
dilation <https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm>`__ is
applied to the “eroded” image, which in theory undoes the erosion except
the bright “features” that were completely eaten away. The overall
effect of this process is to remove any bright feature that is smaller
than a disk with radius ``wnd``. Thus, when setting ``wnd`` to the
expected size of **largest** cell diamater, this process can give us a
good estimation of the background. Pragmatically **10** works well.

For ``method='uniform'``, a `uniform
filter <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.ndimage.uniform_filter.html>`__
(basically a two dimensional rolling mean) is applied to each frame.
``wnd`` controls the window size of the filter, and the result is used
as the background. This is only useful if previous steps failed to
remove some stable, large scale background, and should be less
preferrable than ``"tophat"`` otherwise.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(visualize_preprocess(varr_ref.isel(frame=0), remove_background, method=['tophat'], wnd=[10, 15, 20]))

.. code:: ipython3

    varr_ref = remove_background(varr_ref, **param_background_removal)

save result
-----------

Recall the parameters for ``save_minian``:

.. code:: python

   param_save_minian = {
       'dpath': dpath,
       'fname': 'minian',
       'backend': 'zarr',
       'meta_dict': dict(session_id=-1, session=-2, animal=-3),
       'overwrite': True}

As was mentioned during the **Setting up** step, the ``save_minian``
function decides how your data will be saved: ``dpath`` is the path
under which the actual data file will be stored, and ``fname`` is the
file name of the data file. ``backend`` can be either ``'netcdf'`` or
``'zarr'`` – currently ``'netcdf'`` is more stable and is the
recommended storage option from ``xarray``, but it might suffer
performance issues when running out-of-core computation. ``'zarr'`` is
designed for parallel and out-of-core computation, and is therefore what
is recommended. ``meta_dict`` is a ``dictionary`` that is used to
construct meta data for the final labeled data structure which can be
modified to suit the specific user’s data storing structure.
``overwrite`` is a boolean value (i.e. True/False) controlling whether
the data is overwritten when the file already exists. We set it to
``True`` here so you can easily play with the demo multiple times, but
**use extreme caution** with this during actual analysis – it won’t ask
again for your confirmation.

In particular, here we are saving our minimally-processed video
(``varr_ref``) in ``DataArray`` format. We give it a “name” ``"org"`` by
calling the
`rename <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.rename.html>`__
method on the array, which is ``xarray``\ ’s internal naming system that
stays with the actual data and will be displayed when you print out the
``DataArray``. In practice, give it a name that’s human-readable and be
sure to not name two pieces of data with the same name (Otherwise an
error will occur if you try to combine them in a single dataset).

.. code:: ipython3

    %%time
    varr_ref = varr_ref.chunk(chk)
    varr_ref = save_minian(varr_ref.rename('org'), **param_save_minian)

motion correction
=================

load in from disk
-----------------

Here we load in the data we just saved. We use ``'fname'`` and
``'backend'`` from ``param_save_minian`` since they should be the same
and you don’t have to specify the same information twice.

.. code:: ipython3

    varr_ref = open_minian(dpath,
                          fname=param_save_minian['fname'],
                          backend=param_save_minian['backend'])['org']

estimate shifts
---------------

Recall the parameters for ``estimate shifts``:

.. code:: python

   param_estimate_shift = {
       'dim': 'frame',
       'max_sh': 20}

The idea behind ``estimate_shift_fft`` is simple: for each frame it
calculates a two-dimensional
`cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`__
between that frame and a template frame using
`fft <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`__. The
argument ``'dim'`` specifies along which dimension to run the shift
estimation, and should always be set to ``'frame'`` for this pipeline.
To properly calculate the correlation we have to zero-pad the input
frame, otherwise our estimation will be biased towards zero shifts. The
amount of zero-padding essentially determine the maximum amount of
shifts that can be accounted for, and ``max_sh`` controls this quantity
in pixels. The results from ``estimate_shift_fft`` are saved in a two
dimensional ``DataArray`` called ``shifts``, with two labels on the
``variable`` dimension, representing the shifts along ``'height'`` and
``'width'`` directions.

.. code:: ipython3

    %%time
    shifts = estimate_shifts(varr_ref.sel(subset_mc), **param_estimate_shift)

save shifts
-----------

.. code:: ipython3

    %%time
    shifts = shifts.chunk(dict(frame=chk['frame'])).rename('shifts')
    shifts = save_minian(shifts, **param_save_minian)

visualization of shifts
-----------------------

Here, we visualize ``shifts`` as a fluctuating curve along ``frame``\ s.
This is the first time we explicitly use the package
`holoviews <http://holoviews.org>`__, which is a really nice package for
visualizing data in an interactive manner, and it is highly recommended
that you read through the holoviews tutorial to get familiar with its
syntax.

.. code:: ipython3

    %%opts Curve [frame_width=500, tools=['hover'], aspect=2]
    hv.output(size=output_size)
    if interactive:
        display(hv.NdOverlay(dict(width=hv.Curve(shifts.sel(variable='width')),
                                  height=hv.Curve(shifts.sel(variable='height')))))

apply shifts
------------

After determining what each frame’s shift from the template is, we use
the function ``apply_shifts``, which takes as inputs our video
(``varr_ref``) and (``shifts``) and returns the movie we want (``Y``).
Notably, pixels that are shifted inside the field of view will result in
NaN values (``np.nan``) along the edges of our video, and we have to
decide what to do with these. The default is to fill them with 0.

.. code:: ipython3

    Y = apply_shifts(varr_ref, shifts)
    Y = Y.fillna(0).astype(varr_ref.dtype)

Alternatively you can leverage the
`dropna <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.dropna.html>`__
function to drop them, or
`fillna <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.fillna.html>`__
to fill them with a specific value (potentially ``varr_mc.min()``)

For example, instead of filling the NaN pixels with the nearest
available value, you drop these pixels with the following code:

.. code:: python

   varr_mc = varr_mc.where(varr_mc.isnull().sum('frame') == 0).dropna('height', how='all').dropna('width', how='all')

visualization of motion-correction
----------------------------------

Here we visualize the final result of motion correction (``varr_mc``)
with ``VArrayViewer``. The optional argument ``framerate`` only controls
how the frame slider behaves, not how the data is handled.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(
            [varr_ref.rename('before_mc'), Y.rename('after_mc')],
            framerate=5,
            summary=None,
            layout=True)
        display(vaviewer.show())

save result
-----------

.. code:: ipython3

    %%time
    Y = Y.chunk(chk)
    Y = save_minian(Y.rename('Y'), **param_save_minian)

generate video for motion-correction
------------------------------------

Here we have some additional visualizations for motion correction. We
can generate a video and play it to quickly go through the results. In
addition we can look at the max projection before and after motion
correction. If there were a lot of translational motion presented in the
raw video, we expect the border of cells are much more well-defined, and
even some “different” cells (due to motion) are “merged” together in the
max projection.

.. code:: ipython3

    %%time
    vid_arr = xr.concat([varr_ref, Y], 'width').chunk(dict(height=-1, width=-1))
    vmax = varr_ref.max().compute().values
    write_video(vid_arr / vmax * 255, 'minian_mc.mp4', dpath)

.. code:: ipython3

    im_opts = dict(frame_width=500, aspect=752/480, cmap='Viridis', colorbar=True)
    (regrid(hv.Image(varr_ref.max('frame').compute(), ['width', 'height'], label='before_mc')).opts(**im_opts)
     + regrid(hv.Image(Y.max('frame').compute(), ['width', 'height'], label='after_mc')).opts(**im_opts))

initialization
==============

In order to run CNMF, we first need to generate an initial estimate of
where our cells are likely to be and what their temporal activity is
likely to look like. The whole initialization section is adapted from
the `MIN1PIPE <https://github.com/JinghaoLu/MIN1PIPE>`__ package. See
their
`paper <https://www.cell.com/cell-reports/fulltext/S2211-1247(18)30826-X>`__
for full details about the theory. Here we only give enough information
so that we can select parameters.

load in from disk
-----------------

The first thing we want to do is open up the dataset we just saved.

.. code:: ipython3

    %%time
    minian = open_minian(dpath,
                         fname=param_save_minian['fname'],
                         backend=param_save_minian['backend'])

Here we get the movie (``Y``) from the dataset, calculate a max
projection that will be used later, and generate a flattened version of
our video (``Y_flt``), where the original dimemsions ``'height'`` and
``'width'`` are flattened as one dimension ``spatial``.

.. code:: ipython3

    Y = minian['Y'].astype(np.float)
    max_proj = Y.max('frame').compute()
    Y_flt = Y.stack(spatial=['height', 'width'])

generating over-complete set of seeds
-------------------------------------

The first step is to initialize the **seeds**. Recall the parameters:

.. code:: python

   param_seeds_init = {
       'wnd_size': 2000,
       'method': 'rolling',
       'stp_size': 1000,
       'nchunk': 100,
       'max_wnd': 15,
       'diff_thres': 3}

The idea is that we select some subset of frames, compute a max
projection of those frames, and find the local maxima of that max
projection. We keep repeating this process and putting together all the
local maxima we get along the way until we get an overly-complete set of
local maxima/bright-spots, which are the potential locations of cells.
We call them **seeds**. The assumption here is that the center of cells
are brighter than their surroundings on some, but not necessarily all,
frames. The first and only required argument ``seeds_init`` takes is the
video array we want to process (here, ``Y``). There are four additional
arguments controlling how we subset the frames: ``wnd_size`` controls
the window size of each chunk (*i.e* the number of frames in each
chunk); ``method`` can be either ``'rolling'`` or ``'random'``. For
``method='rolling'``, the moving window will roll along ``frame``,
whereas for ``method='random'``, chunks with ``wnd_size`` number of
frames will be randomly selected; ``stp_size`` is only used if
``method='rolling'``, and is the step-size of the rolling window, or in
other words, the distance between the **center** of each rolling window.
For example, if ``wnd_size=100`` and ``stp_size=200``, the windows will
be as follows: **(0, 100)**, **(200, 300)**, **(400, 500)** *etc.*
Obviously that was a **bad** choice since you probably want the windows
to overlap or you will miss cells. ``nchunk`` is only used if
``method='random'``, and is the number of random chunks we will draw.
Additionally we have two parameters controlling how the local maxima are
found. ``'max_wnd'`` controls the window size within which a single
pixel will be choosen as local maxima. In order to capture cells with
all sizes, we actually find local maximas with different window size and
merge all of them, starting from **2** all the way up to ``'max_wnd'``.
Hence ``'max_wnd'`` should be the radius of the **largest** cell you
want to detect. Finally in order to get rid of local maxima with very
little fluctuation, we set a ``'diff_thres'`` which is the minimal
fluorescent diffrence of a seed across ``frame``\ s. Since the linear
scale of the raw data is preserved, we can set this threshold
emprically.

.. container:: alert alert-info

   The default values of ``seeds_init`` usually work fairly well for a
   dense region like CA1. If you are working with deep brain region with
   sparse cells, try to increase wnd_size and stp_size to make the
   following seeds cleaning steps faster and cleaner.

.. code:: ipython3

    %%time
    seeds = seeds_init(Y, **param_seeds_init)

We can visualize the seeds as points overlaid on top of the ``max_proj``
image. Each white dot is a seed and could potentially be the location of
a cell.

.. code:: ipython3

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds)

peak-noise-ratio refine
-----------------------

We further refine seeds based upon their temporal activity. This
requires that we separate our signal based upon frequency, and this also
brings us to the most powerful and important aspect of this pipeline –
parameter exploring. We are going to take a few example seeds and
separate their activity based upon a few frequencies, and we will then
view the results and select a frequency which we beleive best separates
signal from noise.

This will seem to be the most complicated chunk of code so far, but it
is important to read through, since we will see similar things later and
it is a very powerful piece of code that can help you visualize a lot.
The basic idea is we run some function on a small subset of data using
different parameters within a ``for`` loop, and after that visualize the
results using ``holoviews``. Note that interactive mode needs to be set
as ``True`` for parameter exploring steps like this to work.

The goal of this specific piece of code is to determine the “frequency”
at which we can best seperate our signal from noise, which is an
important parameter used at various places below. We will go line by
line: First we create a ``list`` of frequencies we want to try out –
``noise_freq_list``. The “frequency” values here are a proportion of
your **sampling rate**. Note that if you have temporally downsampled,
the proportion here is relative to the downsampled rate. Then we
randomly select 6 seeds from ``seeds_gmm`` and call them
``example_seeds``, which in turn help us pull out the temporal traces
from the movie ``Y_flt``. The traces of the ``example_seeds`` are
assigned to ``example_trace``. We then create an empty dictionary
``smooth_dict`` to store the resulting visualizations. After
initializing these variables, we use a ``for`` loop to iterate through
``noise_freq_list``, with one of the values from the list as ``freq``
during each iteration. Within the loop, we run ``smooth_sig`` twice on
``example_trace`` with the current ``freq`` we are testing out. The
low-passed result is assigned to ``trace_smth_low``, while the high-pass
result is assigned to ``trace_smth_high``. Then we make sure to actually
carry-out the computation by calling the ``compute`` method on the
resulting ``DataArray``\ s. Finally, we turn the two traces into
visualizations: we construct interactive line plots
(`hv.Curve <http://holoviews.org/reference/elements/bokeh/Curve.html>`__\ s)
from them and put them in a container called a
`hv.HoloMap <http://holoviews.org/reference/containers/bokeh/HoloMap.html>`__.
Again if you are confused about how the visualization works, you can
check out `the
tutorial <http://holoviews.org/getting_started/Introduction.html>`__.
After that we store the whole visualization in ``smooth_dict``, with the
keys being the ``freq`` and values corresponding to the result of this
iteration.

.. container:: alert alert-info

   Here you can edit the values that you want to test in the
   noise_freq_list.

.. code:: ipython3

    %%time
    if interactive:
        noise_freq_list = [0.005, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.45]
        example_seeds = seeds.sample(6, axis='rows')
        example_trace = (Y_flt
                         .sel(spatial=[tuple(hw) for hw in example_seeds[['height', 'width']].values])
                         .assign_coords(spatial=np.arange(6))
                         .rename(dict(spatial='seed')))
        smooth_dict = dict()
        for freq in noise_freq_list:
            trace_smth_low = smooth_sig(example_trace, freq)
            trace_smth_high = smooth_sig(example_trace, freq, btype='high')
            trace_smth_low = trace_smth_low.compute()
            trace_smth_high = trace_smth_high.compute()
            hv_trace = hv.HoloMap({
                'signal': (hv.Dataset(trace_smth_low)
                           .to(hv.Curve, kdims=['frame'])
                           .opts(frame_width=300, aspect=2, ylabel='Signal (A.U.)')),
                'noise': (hv.Dataset(trace_smth_high)
                          .to(hv.Curve, kdims=['frame'])
                          .opts(frame_width=300, aspect=2, ylabel='Signal (A.U.)'))
            }, kdims='trace').collate()
            smooth_dict[freq] = hv_trace

After all the loops are done, we put together a holoviews plot
(``hv.HoloMap``) from ``smooth_dict``, and we specify that we want our
traces to ``overlay`` each other along the ``'trace'`` dimension while
being laid out along the ``'spatial'`` dimension. The result turns into
a nicely animated interactive plot, from which we can determine the
frequency that best separates noise and signal.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        hv_res = (hv.HoloMap(smooth_dict, kdims=['noise_freq']).collate().opts(aspect=2)
                  .overlay('trace').layout('seed').cols(3))
        display(hv_res)

Having determined the frequency that best separates signal from noise,
we move on the next step of seeds refining. Recall the parameters:

.. code:: python

   param_pnr_refine = {
       'noise_freq': 0.06,
       'thres': 1,
       'med_wnd': None}

``pnr_refine`` stands for “peak-to-noise ratio” refine. The “peak” and
“noise” here are defined differently from before. First we
seperate/filter the temporal signal for each seed based on frequency –
the signals composed of the lower half of the frequency are regarded as
**real** signals, while the higher half of the frequencies is presumably
**noise** (“half” being relative to `Nyquist
frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`__). Then we
take the peak-to-valley value (really just **max** minus **min**, or,
`np.ptp <https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.ptp.html>`__)
for both the **real** signal and **noise** signal. Then, “peak-to-noise
ratio” is the ratio between the ``np.ptp`` values of **real** and
**noise** signals. So, the critical assumption here is that real cell
activity is of lower frequency while noise is of a higher frequency, and
they seperate at approximately half the Nyquist frequency, or,
one-fourth of the sampling frequency of the video. Moreover, we don’t
want those “seeds” whose **real** signals are buried in **noise**. If
these assumptions does not suit your recordings - for example, if you
have a really low sampling rate, or if your video are unavoidably noisy
- consider skipping this step. The function ``pnr_refine`` takes in
``varr`` and ``seeds`` as its first two arguments; the ``noise_freq``
that best separates signal and noise, which hopefully has been
determined from the previous cell; and ``thres``, a threshold for
“peak-to-noise ratios” below which seeds will be discarded.
Pragmatically ``thres=1`` works fine and makes sense. You can also use
``thres='auto'``, where a gaussian mixture model with 2 components will
be run on the peak-to-noise ratios and seeds will be selected if they
belong to the “higher” gaussian. ``med_wnd`` is the window size of the
median filter that gets passed in as ``size`` in
```scipy.ndimage.filters.median_filter`` <https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.median_filter.html>`__.
This is only useful in rare cases where the signal of some seeds assume
a huge change in baseline fluorescence and it is not desirable to keep
such seeds. In this case the median-filtered signal is subtracted from
the original signal to get rid of the artifact. In other cases
``'med_wnd'`` should be left to ``None``.

Now we can use the previous visualization result to pick the best
frequency!

|pnr_param|

.. container:: alert alert-success

   What we are looking for here is the frequency that can seperate real
   signal and noise the best, which means the left panel in the example
   trace, with the ``noise_freq`` = 0.005, is not ideal. In the mean
   time, we also don’t want the signal bands to be overly thick which is
   showing in the right panel with the ``noise_freq`` = 0.45. Thus, the
   middle trace with ``noise_freq`` = 0.05 best suits the needs!

.. container:: alert alert-info

   Now, say you already found your parameters, it’s time now to pass
   them in! Either go back to initial parameters setting step and modify
   them there, or call the parameter here and change its value/s
   accordingly.

For example, if you want to change ``noise_freq`` to 0.05, and start
using median filter equal to 501 here:

.. code:: python

   param_pnr_refine['noise_freq'] = 0.05
   param_pnr_refine['med_wnd'] = 501

Finally, run the following code cell to further clean the seeds:

.. |pnr_param| image:: img/pnr_param_v2.png

.. code:: ipython3

    seeds, pnr, gmm = pnr_refine(Y_flt, seeds.copy(), **param_pnr_refine)

Here in the belowing code cell we will visualize the gmm fit, but
**only** when you chose ``thres='auto'`` before. The x axis here is pnr
ratio value, and the x value of the intersection of blue and red curve
is the auto chose threshold, everything below this threshold will be
seen as noise.

.. code:: ipython3

    if gmm:
        display(visualize_gmm_fit(pnr, gmm, 100))

And again we can visualize seeds that’s taken out during this step.

.. code:: ipython3

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds, 'mask_pnr')

Still, white dots are accepted seeds and red dots are taken out.

.. container:: alert alert-info

   if you see seeds that you believe should be cells have been taken out
   here, either skip this step or try lower the threshold a bit. You can
   also use the individual trace ploting method we discussed at the end
   of gmm_refine part to look into specific seed.

ks refine
---------

``ks_refine`` refines the seeds using `Kolmogorov-Smirnov
test <https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test>`__. Recall
the parameters:

.. code:: python

   param_ks_refine = {
       'sig': 0.05}

The idea is simple: if a seed corresponds to a cell, its fluorescence
intensity across frames should be somewhat
`bimodal <https://en.wikipedia.org/wiki/Multimodal_distribution>`__,
with a large normal distribution representing silence/little activity,
and another peak representing when the seed/cell is active. Thus, we can
carry out KS test on the intensity distribution of each seed, and keep
only the seeds where the null hypothesis (that the fluoresence is simply
a normal distribution) is rejected. ``ks_refine`` takes in ``varr`` and
``seeds`` as its first two arguments, then a ``sig`` which is the
significance level at which the null hypothesis is rejected (defaulted
to **0.05**).

.. container:: alert alert-info

   In practice, we have found this step tends to take away real cells
   when video are very short (for example, the one that comes with this
   package under “./demo_movies”). This is likely because the number of
   “active” frames is too small. Feel free to skip this step if you
   encounter the same situation.

.. code:: ipython3

    %%time
    seeds = ks_refine(Y_flt, seeds[seeds['mask_pnr']], **param_ks_refine)

.. code:: ipython3

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds, 'mask_ks')

merge seeds
-----------

At this point, much of our refined seeds likely reflect the position of
an actual cell. However, we are likely to still have multiple seeds per
cell, which we want to avoid. Here we discard redudant seeds through a
process of merging.

Recall the parameters:

.. code:: python

   param_seeds_merge = {
       'thres_dist': 5,
       'thres_corr': 0.7,
       'noise_freq': .06'}

The function ``seeds_merge`` attempts to merge seeds together which
potentially come from the same cell, based upon their spatial distance
and temporal correlation. Specifically, ``thres_dist`` is the threshold
for euclidean distance between pairs of seeds, in pixels, and
``thres_corr`` is the threshold for pearson correlation between pairs of
seeds. In addition, it’s very beneficial to smooth the signals before
running the correlation, and again ``noise_freq`` determines how
smoothing should be done. In addition to feeding in a number, such as
the noise frequency you defined earlier during ``seeds_refine_pnr``, you
can also use ``noise_freq='envelope'``. When ``noise_freq='envelope'``,
a hilbert transform will be run on the temporal traces of each seed and
the correlation will be calculated on the envelope signal. Any pair of
seeds that are within ``thres_dist`` **and** has a correlation higher
than ``thres_corr`` will be merged together, such that only the seed
with maximum intensity in the max projection of the video will be kept.
Thus ``thres_dist`` should be the expected size of cells and
``thres_corr`` should be relatively high to avoid over-merging.

.. container:: alert alert-info

   Potentially we could pick out multiple seeds that are actually within
   one cell, but we want to avoid that as much as possible to have a
   clean start for CNMF later, you can try lower the thres_corr or raise
   up the thres_dist to merge more cells. Ideally, you want to see only
   one accepted seed (white dot) within each cell.

.. code:: ipython3

    %%time
    seeds_final = seeds[seeds['mask_ks']].reset_index(drop=True)
    seeds_mrg = seeds_merge(Y_flt, seeds_final, **param_seeds_merge)

.. code:: ipython3

    hv.output(size=output_size)
    visualize_seeds(max_proj, seeds_mrg, 'mask_mrg')

initialize spatial and temporal matrices from seeds
---------------------------------------------------

Up till now, the seeds we have are only one-pixel dots. In order to kick
start CNMF we need something more like the spatial footprint (``A``) and
temporal activities (``C``) of real cells. Thus we need to
``initilalize`` ``A`` and ``C`` from the seeds we have (``seeds_mrg``).
Recall the parameters:

.. code:: python

   param_initialize = {
       'thres_corr': 0.8,
       'wnd': 10,
       'noise_freq': .06
   }

To obtain the initial spatial matrix ``A``, for each seed, we simply use
a Pearson correlation between the seed and surrounding pixels.
Apparantly cacluating correlation with all other pixels for every seed
is time-consuming and unnecessary. ``'wnd'`` controls the window size
for calculating the correlation, and thus is the maximum possible size
of any spatial footprint in the initial spatial matrix. At the same time
we do not want pixels with low correlation value to influence our
estimation of temporal signals, thus a ``'thres_corr'`` is also
implemented where only pixels with correlation above this threshold are
kept. After generating ``A``, for each seed, we calculate a weighted
average of pixels around the seed, where the weight are the initial
spatial footprints in ``A`` we just generated. We use this weighted
average as the initial estimation of temporal activities for each units
in ``C``. Finally, we need two more terms: ``b`` and ``f``, representing
the spatial footprint and temporal dynamics of the **background**,
respectively. Since usually the backgrounds are already removed at this
stage, we provide a very simple estimation of remaining background – we
simply mask ``Y`` with the spatial footprints of units in ``A``, that
is, we only keep pixels that does not appear in the spatial foorprints
of any units. We calculate a mean projection across ``frame``\ s and use
as ``b``, and we calculate mean fluorescence along ``frame``\ s and use
as ``f``.

.. code:: ipython3

    %%time
    A, C, b, f = initialize(Y, seeds_mrg[seeds_mrg['mask_mrg']], **param_initialize)

Finally we visualize the result of our initialization by plotting a
projection of the spatial matrix ``A``, a raster of the temporal matrix
``C``, as well as background terms ``b`` and ``f``.

.. code:: ipython3

    im_opts = dict(frame_width=500, aspect=A.sizes['width']/A.sizes['height'], cmap='Viridis', colorbar=True)
    cr_opts = dict(frame_width=750, aspect=1.5*A.sizes['width']/A.sizes['height'])
    (regrid(hv.Image(A.sum('unit_id').rename('A').compute(), kdims=['width', 'height'])).opts(**im_opts)
     + regrid(hv.Image(C.rename('C').compute(), kdims=['frame', 'unit_id'])).opts(cmap='viridis', colorbar=True, **cr_opts)
      + regrid(hv.Image(b.rename('b').compute(), kdims=['width', 'height'])).opts(**im_opts)
     + datashade(hv.Curve(f.rename('f').compute(), kdims=['frame']), min_alpha=200).opts(**cr_opts)
    ).cols(2)

save results
------------

Then we save the results in the dataset. Note here that we change the
name of a dimension by writing ``rename(unit_id='unit_id_init')``. The
name of this dimension is changed as a precaution, since the size of the
dimension ``unit_id`` will likely change in the next section **CNMF**.
During CNMF, most likely units will be merged, and there will be
conflicts if we save other variables with dimension ``unit_id`` that
have different coordinates.

.. code:: ipython3

    %%time
    A = save_minian(A.rename('A_init').rename(unit_id='unit_id_init'), **param_save_minian)
    C = save_minian(C.rename('C_init').rename(unit_id='unit_id_init'), **param_save_minian)
    b = save_minian(b.rename('b_init'), **param_save_minian)
    f = save_minian(f.rename('f_init'), **param_save_minian)

CNMF
====

This section assume you already have some knowledge about using CNMF as
a method of extracting neural activities from video. If not, it is
recommended that you read `the
paper <https://www.sciencedirect.com/science/article/pii/S0896627315010843>`__,
to get a broad understanding of the problem and proposed solution.

As a quick reminder, here is the essential idea of CNMF: We believe our
movie, ``Y``, with dimensions ``height``, ``width`` and ``frame``, can
be written in (and thus broken down as) the following equation:

.. math:: \mathbf{Y} = \mathbf{A} \cdot \mathbf{C} + \mathbf{b} \cdot \mathbf{f} + \epsilon

\ where ``A`` is the spatial footprint of each unit, with dimension
``height``, ``width`` and ``unit_id``; ``C`` is the temporal activities
of each unit, with dimension ``unit_id`` and ``frame``; ``b`` and ``f``
are the spatial footprint and temporal activities of some background,
respectively; and :math:`\epsilon` is the noise. Note that strictly
speaking, matrix multiplication is usually only defined for two
dimensional matrices, but our ``A`` here has three dimensions, so in
fact we are taking the `tensor
product <https://en.wikipedia.org/wiki/Tensor_product>`__ of ``A`` and
``C``, reducing the dimension ``unit_id``. This might seem to complicate
things (compared to just treating ``height`` and ``width`` as one
flattened ``spatial`` dimension), but it ends up making some sense. When
you take a dot product of any two “matrices” on a certain **dimension**,
all that is happening is a **product** followed by a **sum** – you take
the product for all pairs of matching numbers coming from the two
“matrices”, where “match” is defined by their index along said
dimension, and then you take the sum of all those products along the
dimension. Thus when we take the tensor product of ``A`` and ``C``, we
are actually multiplying all those numbers in dimension ``height``,
``width`` and ``frame``, matched by ``unit_id``, and then take the sum.
Conceptually, for each unit, we are weighting the spatial footprint
(``height`` and ``width``) by the fluorecense of that unit on given
``frame``, which is the **product**, and then we are overlaying all
units together, which is the **sum**. With that, the equation above is
trying to say that our movie is made up of a weighted sum of the spatial
footprint and temporal activities of all units, plus some background and
noise.

Now, there is another rule about ``C`` that separates it from background
and noise, and saves it from being just some random matrix that happens
to fit well with the data (``Y``) without having any biological meaning.
This rule is the second essential idea of CNMF: each “row” of ``C``,
which is the temporal trace for each unit, should be described as an
`autoregressive
process <https://en.wikipedia.org/wiki/Autoregressive_model>`__ (AR
process), with a parameter ``p`` defining the **order** of the AR
process:

.. math::  c(t) = \sum_{i=0}^{p}\gamma_i c(t-i) + s(t) + \epsilon

\ where :math:`c(t)` is the calcium concentration at time (``frame``)
:math:`t`, :math:`s(t)` is spike/firing rate at time :math:`t` (what we
actually care about), and :math:`\epsilon` is noise. Basically, this
equation is trying to say that at any given time :math:`t`, the calcium
concentration at that moment :math:`c(t)` depends on the spike at that
moment :math:`s(t)`, as well as its own history up to ``p`` time-steps
back :math:`c(t-i)`, scaled by some parameters :math:`\gamma_i`\ s, plus
some noise :math:`\epsilon`. Another intuition of this equation comes
from looking at different ``p``\ s: when ``p=0``, the calcium
concentration is an exact copy of the spiking activities, which is
probably not true; when ``p=1``, the calcium concentration has an
instant rise in response to a spike followed by an exponential decay;
when ``p=2``, calcium concentration has some rise time following a spike
and an exponential decay; when ``p>2``, more convoluted waveforms start
to emerge.

With all this in mind, CNMF tries to find the spatial matrix (``A``) and
temporal activity (``C``) (along with ``b`` and ``f``) that best
describe ``Y``. There are a few more important practical concerns:
Firstly we cannot solve this problem in one shot – we need to
iteratively and separately update ``A`` and ``C`` to approach the true
solution – and we need something to start with (that is what
**initilization** section is about). Surprisingly often times 2
iterative steps after our initialization seem to give good enough
results, but you can always add more iterations (and you should be able
to easily do that after reading the comments). Secondly, by intuition
you may define “best describe ``Y``” as the results that minimize the
noise :math:`\epsilon` (or residuals, if you will). However we have to
control for the
`sparsity <https://en.wikipedia.org/wiki/Sparse_matrix>`__ of our model
as well, since we do not want every little random pixel that happens to
correlate with a cell to be counted as part of the spatial footprint of
the cell (non-sparse ``A``), nor do we want a tiny spike at every frame
trying to explain every noisy peak we observe (non-sparse ``C``). Thus,
the balance between fidelity (minimizing error) and sparsity (minimizing
non-zero entries) is an important idea for both the spatial and temporal
update.

loading data
------------

First we load in our data from previous steps. ``'unit_id'`` is renamed
as a precaution, mentioned at the end of the **initialization** section.

.. code:: ipython3

    %%time
    minian = open_minian(dpath,
                         fname=param_save_minian['fname'],
                         backend=param_save_minian['backend'])
    Y = minian['Y'].astype(np.float)
    A_init = minian['A_init'].rename(unit_id_init='unit_id')
    C_init = minian['C_init'].rename(unit_id_init='unit_id')
    b_init = minian['b_init']
    f_init = minian['f_init']

estimate spatial noise
----------------------

Prior to performing CNMF’s first spatial update, we need to get a sense
of how much noise is expected, which we will then feed into CNMF. To do
so, we compute an fft-transform for every pixel independently, and
estimate noise from its `power spectral
density <https://en.wikipedia.org/wiki/Spectral_density>`__. Recall the
parameters:

.. code:: python

   param_get_noise = {
       'noise_range': (0.06, 0.5),
       'noise_method': 'logmexp'}

Note that the number in ``noise_range`` is relative to the sampling
frequency, so **0.5** actually represents the Nyquist frequency and is
the highest you can go as far as fft is concerned. Thus **(0.25, 0.5)**
is the higher frequency half of the signal. After choosing
``noise_range``, we have to decide how to collapse across different
frequencies to get a single number of noise power for each pixel. Three
``noise_method``\ s are availabe: ``noise_method='mean'`` and
``noise_method='median'`` will use the mean and median across all
``freq`` as the estimation of noise for each pixel.
``noise_method='logmexp'``\ is a bit more complicated – the equation is
as follows: :math:`sn = \exp( \operatorname{\mathbb{E}}[\log psd] )`
where :math:`\exp` is the `exponential
function <Exponential_function>`__, :math:`\operatorname{\mathbb{E}}` is
the `expectation
operator <https://en.wikipedia.org/wiki/Expected_value>`__ (mean),
:math:`\log` is `natural
logarithm <https://en.wikipedia.org/wiki/Natural_logarithm>`__,
:math:`psd` is the spectral density of noise for any pixel, and
:math:`sn` is the resulting estimation of noise power. It is recommended
to keep ``noise_method='logmexp'`` since this is the default behavior of
the `CaImAn <https://github.com/flatironinstitute/CaImAn>`__ package.

.. container:: alert alert-info

   In order to define the lower bound of noise_range (the upper bound
   can be left equal to 0.5), examine the PSD plot and define the
   frequency value (again, this is actually a proportion of your
   sampling rate), where power has dropped off across all pixels (i.e.,
   spatial).

.. code:: ipython3

    %%time
    sn_spatial = get_noise_fft(Y, **param_get_noise).persist()

test parameters for spatial update
----------------------------------

We will now do some parameter exploring before actually performing the
first spatial update. We do this because we do not want to do a
10-minute spatial update only to find the selected parameters do not
produce nice results. For parameter exploration, we will analyze a very
small subset of data so that we can quickly examine the influence of
various paramater values. Here, we randomly select 10 units from
``A_init.coords['unit_id']`` with the help of
```np.random.choice`` <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html>`__.

.. code:: ipython3

    if interactive:
        units = np.random.choice(A_init.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_init.sel(unit_id=units).persist()
        C_sub = C_init.sel(unit_id=units).persist()

Here, we again perform parameter exploration using a ``for`` loop and
visualization with help of ``dict`` and ``holoviews``, only this time we
use a convenient function, ``visualize_spatial_update`` from ``minian``,
to handle all the visualization details. For now, the sparseness penalty
(``sparse_penal``) is only one parameter in ``update_spatial`` that we
are interested in playing with, but there is nothing stopping you from
adding more. Discussion of all the parameters for ``update_spatial``
will follow soon.

.. container:: alert alert-info

   Here, you can simply add the values that you want to test or delete
   the values you are not interested in from spar_ls. Pragmatically, the
   range of 0.05 to 1 is reasonable.

.. code:: ipython3

    %%time
    if interactive:
        sprs_ls = [0.05, 0.1, 0.5]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_b, cur_C, cur_f = update_spatial(
                Y, A_sub, b_init, C_sub, f_init,
                sn_spatial, dl_wnd=param_first_spatial['dl_wnd'], sparse_penal=cur_sprs)
            if cur_A.sizes['unit_id']:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = cur_C.compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=['sparse penalty'])

Finally, we actually plot the visualization ``hv_res``. What you should
expect here will be explained later along with what ``sparse_penal``
actually does.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

first spatial update
--------------------

Here is the idea behind ``update_spatial``. Recall the parameters:

.. code:: python

   param_first_spatial = {
       'dl_wnd': 10,
       'sparse_penal': 0.01,
       'update_background': True,
       'normalize': True,
       'zero_thres': 'eps'}

To reiterate, the big picture is that given the data (``Y``) and our
units’ activity (``C``) from previous the update (which is ``C_init``),
we want to find the spatial footprints (``A``) such that 1. the
**error** ``Y - A.dot(C, 'unit_id')`` is as small as possible, and 2.
the `l1-norm <http://mathworld.wolfram.com/L1-Norm.html>`__ of ``A`` is
as small as possible. Here the **l1-norm** is a proxy to control for the
sparsity of ``A``. Ideally to promote sparsity we want to control for
the number of non-zero entries in ``A``, which is the
`l0-norm <https://en.wikipedia.org/wiki/Lp_space#When_p_=_0>`__. However
optimizing for the l0-norm is typically `computationally hard to
do <https://stats.stackexchange.com/questions/269298/why-do-we-only-see-l-1-and-l-2-regularization-but-not-other-norms>`__,
and it is usually good enough to use **l1-norm** instead as a proxy.

Now, in theory we want to update every entry in ``A`` iteratively with
the above two goals in mind. However, updating that amount of numbers in
``A`` is still computationally very demanding, and it is much better if
we can breakdown our big problem into smaller chunks that can be
parallelized (making things much faster). **CNMF** is all about solving
the issues caused by overlapping neurons, so it is best to keep the
dependency across units (along dimension ``unit_id``) and update these
entries together. However, it should be fine to treat each pixel as
independent and update different pixels separately (in parallel). Thus,
our new, “smaller” problem is: for each pixel, find the corresponding
pixel in ``A``, across all ``unit_id``, that give us smallest
**l1-norm** as well as smallest **error** when multiplied by ``C``. In
equation form, this is:

.. math::

   \begin{equation*}
   \begin{aligned}
   & \underset{A_{ij}}{\text{minimize}}
   & & \left \lVert Y_{ij} - A_{ij} \cdot C \right \rVert ^2 + \alpha \left \lvert A_{ij} \right \rvert \\
   & \text{subject to}
   & & A_{ij} \geq 0 
   \end{aligned}
   \end{equation*}

where we use :math:`A_{ij}` to represent one pixel in ``A``, like
``A.sel(height=i, width=j)``, which will only have one dimension left:
``unit_id``. Similarly :math:`Y_{ij}` is the corresponding pixel in
``Y`` which will only have the dimension ``frame`` left. Thus,
:math:`\left \lVert Y_{ij} - A_{ij} \cdot C \right \rVert ^2` is our
**error** term and :math:`\left \lvert A_{ij} \right \rvert` is our
**l1-norm**. Moreover, we put these two terms together as a unitary
target function/common goal to be minimized, with :math:`\alpha`
controlling the balance between them. This balance can be seen by
considering the impact of :math:`\alpha`: the higher the value of
:math:`\alpha`, the greater the contribution the **l1-norm** term makes
to the common goal (target function), the more penalty/emphasis you
place on sparsity, and as a result, the more sparse ``A`` will be. The
determination of the exact value of :math:`\alpha` is rather
complicated, but the parameter we have for ``update_spatial`` is
relative, where ``alpha=1`` corresponds to the default behavior of
**CaImAn** package, and is usually a good place to start testing.

.. container:: alert alert-success

   Here is a good place to bring back the parameter exploring
   visualization results from the previous step and make sense of them!
   Pragmatically, relatively small values of sparse_penal have very
   little impact on the resulting A, but once you hit a large enough
   value, you will start to see units getting dimmer, sometimes
   completely disappearing. You might think this is the sparsity penalty
   in action, but from experience this is usually a case you want to
   avoid. After all, update_spatial has no way to differentiate noise
   from cells other than their corresponding temporal trace. Thus, you
   do not want update_spatial to take out cells for you unless you
   strongly trust the temporal traces (which you shouldn’t for now since
   it’s the first update and the temporal traces we have are merely
   weighted means of the original movie). If you are still puzzled about
   how to pick the right sparse_panel from the previous parameter
   exploring step, below we provide an illustrative example.

.. figure:: img/sparse_panel_spatial_update.PNG
   :alt: 1st spatial update param exploring

   1st spatial update param exploring

.. container:: alert alert-success

   What you are seeing here is parameter testing of the first spatial
   update. The left panel is the result with sparse_penal = 0.01, the
   middle panel the results with sparse_penal = 0.3, and the right the
   results with sparse_penal = 1. Ideally, we want the Binary Spatial
   Matrix to best mimic the real spatial footprint, which also means,
   they should be shaped like a cell. Thus, in this specific example,
   sparse_panel = 0.01 (left penal) is not a good choice. Secondly, we
   also don’t want to actually get rid of cells by using a high sparse
   panelty value at this step, which means sparse_panel = 1 (right
   penal) is not good as well. Thus, sparse_panel = 0.3 (middle panel)
   is a fairly good parameter to choose here.

There is yet another parameter, ``dl_wnd``, that is relevant to
practical consideration. Recall that we are updating :math:`A_{ij}` for
our “small” problem, which has the dimension ``unit_id`` and has
``A.sizes['unit_id]`` number of entries (that is, the number of units).
This is computationally feasible, but still a lot, especially when you
do this for all pixels. One way to reduce computational demand is to
leave out certain units when updating certain pixels – in particular, it
does not make sense to consider a unit that is supposed to be at the top
left corner of the field of view when we update a pixel in the bottom
right corner. In other words, for each pixel, we solve the “small”
problem with only a subset of all potential units, thus hugely
increasing the speed of ``update_spatial``. This is where ``A_init``
comes into play (actually the only place it is used – we do not need
``A`` at all for the update itself). We compute a morphological
dilation, like that used during `background
removal <#background-removal>`__, on ``A_init``, unit by unit, with
window size ``dl_wnd``, and we use the result as a **masking matrix**.
Then, during the actual update of any given pixel, only units that have
a non-zero value at the corresponding pixel in the **masking matrix**
will be considered for update. In other words, we are allowing each unit
to expand from ``A_init`` up to a distance of ``dl_wnd``, and killing
off any possibility beyond that range. The rationale of using ``dl_wnd``
here is that even if for some reason we have only one non-zero pixel
representing the center of a certain unit in ``A_init``, that unit can
potentially expand to a full size cell, but anything beyond that would
probably be either part of other cells or random noise. Thus, we want to
set ``dl_wnd``\ to approximately the radius of the largest cell to help
ensure we get a clean footprint for all cells.

Then we have a boolean parameter, ``update_background``, controlling
whether we want to update the background in this step. This is the only
place in the pipeline that the background will be updated, and the way
it is updated is by essentially treating ``b`` as another ``unit`` and
updating it according to the temporal activity ``f``. Pragmatically
since the morphology-based `background removal <#backgroun-removal>`__
works so well at cleaning the backgrounds, this updating has little
impact on the result.

Due to the actual implementation of the optimization method, it is hard
for the computer to set some variables to absolutely zero. Instead, we
usually have a very small float numbers in place of zeros.
``zero_thres`` solves this by thresholding all the values and setting
anything below ``zero_thres`` to zero. You want to use a very small
number for ``zero_thres``. Setting ``zero_thres='eps'`` will use the
`machine
epsilon <https://en.wikipedia.org/wiki/Machine_epsilon>`__\ (the
smallest non-negative number a machine can represent) of current
datatype.

Finally, we have an additional step after everything: normalization so
that the spatial footprint of each unit has unit-norm. In practice we
found that normalizing the result helps promoting the numerical
stability of the algorithm, and enable us to interpret the spatial
footprints as “weights” on each pixel so that the temporal activities
are in the same scale space across units and can be compared. However
normlizing spatial footprint for each unit does not preserve the
relationship between overlapping cells in terms of their relative
contribution to the activities of shared pixels. If such interpretation
is critical for your downstream analysis, consider turning this off.

``update_spatial`` takes in the original data (``Y``), the initial
spatial footprint for units and background (``A`` and ``b``,
respectively), the initial temporal trace for units and background
(``C`` and ``f``, respectively), and the estimated noise on each pixel
(``sn``), in that order. Optional arguments are ``sparse_penal``,
``dl_wnd``, ``update_background``, ``post_scal`` and ``zero_thres``.

.. code:: ipython3

    %%time
    A_spatial, b_spatial, C_spatial, f_spatial = update_spatial(
        Y, A_init, b_init, C_init, f_init, sn_spatial, **param_first_spatial)

.. code:: ipython3

    hv.output(size=output_size)
    opts = dict(plot=dict(height=A_init.sizes['height'], width=A_init.sizes['width'], colorbar=True), style=dict(cmap='Viridis'))
    (regrid(hv.Image(A_init.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints Initial")
    + regrid(hv.Image((A_init.fillna(0) > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints Initial")
    + regrid(hv.Image(A_spatial.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints First Update")
    + regrid(hv.Image((A_spatial > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints First Update")).cols(2)

.. code:: ipython3

    hv.output(size=output_size)
    opts_im = dict(plot=dict(height=b_init.sizes['height'], width=b_init.sizes['width'], colorbar=True), style=dict(cmap='Viridis'))
    opts_cr = dict(plot=dict(height=b_init.sizes['height'], width=b_init.sizes['height'] * 2))
    (regrid(hv.Image(b_init.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial Initial')
     + datashade(hv.Curve(f_init.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal Initial')
     + regrid(hv.Image(b_spatial.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial First Update')
     + datashade(hv.Curve(f_spatial.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal First Update')
    ).cols(2)

test parameters for temporal update
-----------------------------------

First off we select some ``units`` to do parameter exploring.

.. code:: ipython3

    if interactive:
        units = np.random.choice(A_spatial.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_spatial.sel(unit_id=units).persist()
        C_sub = C_spatial.sel(unit_id=units).persist()

Now we move on to the parameter exploring of temporal update. Here we
use the same idea we have before, only this time there is much more
parameters to play with for temporal update, and we now have four
``list``\ s of potential parameters: ``p_ls``, ``sprs_ls``, ``add_ls``,
and ``noise_ls``. We use
```itertools.product`` <https://docs.python.org/3.7/library/itertools.html#itertools.product>`__
to iterate through all possible combinations of the potential values and
save us from nested ``for`` loops.

.. code:: ipython3

    %%time
    if interactive:
        p_ls = [1]
        sprs_ls = [0.01, 0.05, 0.1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = compute_trace(Y, A_sub, b_spatial, C_sub, f_spatial).persist()
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(p_ls, sprs_ls, add_ls, noise_ls):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print("p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}"
                  .format(cur_p, cur_sprs, cur_add, cur_noise))
            YrA, cur_C, cur_S, cur_B, cur_C0, cur_sig, cur_g, cur_scal = update_temporal(
                Y, A_sub, b_spatial, C_sub, f_spatial, sn_spatial, YrA=YrA,
                sparse_penal=cur_sprs, p=cur_p, use_spatial=False, use_smooth=True,
                add_lag = cur_add, noise_freq=cur_noise)
            YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (
                YrA.compute(), cur_C.compute(), cur_S.compute(), cur_g.compute(), cur_sig.compute(), A_sub.compute())
        hv_res = visualize_temporal_update(
            YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict,
            kdims=['p', 'sparse penalty', 'additional lag', 'noise frequency'])

A piece of useful infomation after you run this cell is that under what
testing parameter, which sample units got dropped because of poor fit:
|dropped sample units|

.. container:: alert alert-success

   Cross compare this with the raw trace plot, find the most reasonable
   parameters that drop the right sample cells.

Then, we plot the visualization ``hv_res`` of the 10 ramdom units we
just generated at the belowing code cell. Don’t worry if each parameter
doesn’t make much sense now, What you should expect here will be
explained later in first temporal update along with what
``param_first_temporal`` actually does (Look for the green tips box)!

.. |dropped sample units| image:: img/first_tem_drop_v2.PNG

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

first temporal update
---------------------

Here is the idea for temporal update: Recall tha parameters:

.. code:: python

   param_first_temporal = {
       'noise_freq': 0.06,
       'sparse_penal': 0.1,
       'p': 1,
       'add_lag': 20,
       'use_spatial': False,
       'jac_thres': 0.2,
       'zero_thres': 1e-8,
       'max_iters': 200,
       'use_smooth': True,
       'scs_fallback': False,
       'post_scal': True}

Similar to the spatial update, given the spatial footprint of each unit
(``A``), our goal is now to find the activity of each unit (``C``) that
minimizes both the **error** (``Y - A.dot(C, 'unit_id')``) and the
**l1-norm** of ``C``. However there is an additional constraint: the
trace of each unit in ``C`` must follow an autoregressive process. Due
to this additional layer of complexity, things becomes more
computationaly expensive. To reduce computatioinal cost, first observe
that ``A`` is usually much larger than ``C`` (you usually have more
total pixels than ``frame``\ s), and performing the dot product,
``A.dot(C, 'unit_id')``, everytime you try a different number in ``C``,
is infeasible. Thus, we convert our **error** term to something like
:math:`\mathbf{A}^{-1} \cdot \mathbf{Y} - \mathbf{C}`, where
:math:`\mathbf{A}^{-1}` represents a matrix that can “undo” what ``A``
usually does to ``C`` – instead of weighting the temporal activity of
each unit by its spatial footprint (converting a matrix with dimension
``unit_id`` and ``frame`` into one with dimensions ``height``, ``width``
and ``frame``), :math:`\mathbf{A}^{-1}` “extracts” the temporal activity
of each unit based upon their spatial footprint (converting a matrix
with dimension ``height``, ``width`` and ``frame`` into one with
dimensions ``unit_id`` and ``frame``). In other words,
:math:`\mathbf{A}^{-1}` is like an
`inverse <https://en.wikipedia.org/wiki/Moore–Penrose_inverse>`__ of
``A``. This way, we only need to calculate
:math:`\mathbf{A}^{-1} \cdot \mathbf{Y}` once and be done – we can use
that result everytime we update ``C``. The calculation of
:math:`\mathbf{A}^{-1} \cdot \mathbf{Y}` is rather complicated and not
strictly mathematically accurate, but it provides a good approximation
with huge computational benefit, and is the default behavior of CaImAn.
You can turn this off by supplying ``use_spatial=True`` – however that
is usually too computationally demanding to do. We will assume
``use_spatial=False`` in the following discussion and call the
:math:`\mathbf{A}^{-1} \cdot \mathbf{Y}` term ``YrA``, as in the code.
The second thing to observe is that we cannot keep the ``unit_id``
dimension and chop up the ``frame`` dimension for parallel processing
(like how we chopped up pixels during the spatial update), since we have
to check whether each trace along the ``frame`` dimension follows an
autoregressive process. Instead, we turn to the ``unit_id`` dimension to
make our problem “smaller”. Since we have a relatively good ``A`` now,
it should be OK to update units that are not spatially overlapping
independently. This idea should work if you have a relatively sparse
distribution of cells. However if your field-of-view is packed with
cells, if we were to consider cells overlapping if they share only one
pixel, we would likely end up having to update ``C`` altogether, since
every cell is transitively overlapping with every other cell. Instead,
we put a threshold on how we define “overlap”, and that is what
``jac_thres`` is for – only cells that have an area of their spatial
footprint overlapping that is more than this threshold (ranging from 0
to 1) will be considered “overlapping”. (The “proportion of overlapping
area” has a formal name: `Jaccard
index <https://en.wikipedia.org/wiki/Jaccard_index>`__, hence the name
``jac_thres``). Pragamatically ``jac_thres=0.2`` works for data that is
very compact in cells.

We now turn to the “other layer of complexity,” which is the
autoregressive process. Recall that the temporal trace of each unit
should be fitted by the following equation:

.. math:: c(t) = \sum_{i=0}^{p}\gamma_i c(t-i) + s(t) + \epsilon

\ The first thing we want to determine is ``p``. As discussed before,
``p=2`` is a good choice if your calcium transients have an observable
rise-time. ``p=1`` might work better if the rise-time of your signal is
faster than your sampling rate and you thus don’t need to explicitly
model it. Notably, ``p>2`` could result in
`over-fitting <https://en.wikipedia.org/wiki/Overfitting>`__ and is not
recomended unless you are certain that your calcium traces have a more
complicated waveform. Next, notice that we have several
:math:`\gamma_i`\ s unaccounted for (though usually not too many if
``p`` is small). Luckily, we do not have to iteratively update these –
it turns out that the :math:`\gamma_i`\ s of an autoregressive process
are related to the
`autocovariance <https://en.wikipedia.org/wiki/Autocovariance>`__ of the
signal at different lags, which can be readily computed from ``YrA``.
For full derivation of these relationships, please refer to the
`original CNMF
paper <https://www.sciencedirect.com/science/article/pii/S0896627315010843?via%3Dihub>`__.
Here, we will merely assume that the parameters that affect how much a
signal depends on its own history are related to the covariance of the
signal when you shift it by different temporal lags. In this way,
:math:`\gamma_i`\ s can be computed rather deterministicly. Say you set
``p=2`` and thus you have two :math:`\gamma_i`\ s to be estimated – you
would need exactly two equations involving the autocovariance function
up to 2 time-step lags to give you the two :math:`\gamma_i`\ s. However,
you can add additional equations using different lags to better model
the propogation of signal, since the impact of :math:`\gamma_i`\ s can
theoretically extend infinitely back in time, and should be reflected in
the autocovariance function at any additional lag. In practice, we use a
finite number of equations, solved with `least
squares <https://en.wikipedia.org/wiki/Least_squares>`__. Thus it is
important to choose an appropriate number of **additional** equations,
which is what ``add_lag`` controls. An ``add_lag`` that is too small
like ``add_lag=0`` will leave everything to the first ``p`` number of
equations and autocovariance functions, which might not be reliable.
Pragmatically, smaller ``add_lag`` values tend to bias the
:math:`\gamma_i`\ s to give a much faster decay, whereas larger
``add_lag`` values tend to give a longer decay. **As a rule of thumb, it
is usually good to set ``add_lag`` to approximately the decay time of
your signal (in frames).**

Once we have estimated the :math:`\gamma_i`\ s, the calcium traces,
:math:`c(t)`, and spikes, :math:`s(t)`, are essentially **one thing** –
given calcium traces and how they rise/decay in response to spikes, we
can deduce where the spikes happen, and *vice versa*. We can express
this determined relationship with a matrix :math:`\mathbf{G}` where
:math:`s(t) = \mathbf{G} \cdot c(t)`. In other words, :math:`\mathbf{G}`
is the matrix that “undoes” what :math:`\gamma_i`\ s do to :math:`s(t)`.
With all these parameters sorted out, we finally come to the actual
optimization problem:

.. math::

   \begin{equation*}
   \begin{aligned}
   & \underset{C_{i}}{\text{minimize}}
   & & \left \lVert \mathbf{YrA}_{i} - \mathbf{C}_{i} \right \rVert ^2 + \alpha \left \lvert \mathbf{G}_{i} \cdot \mathbf{C}_{i} \right \rvert \\
   & \text{subject to}
   & & \mathbf{C}_{i} \geq 0, \; \mathbf{G}_{i} \cdot \mathbf{C}_{i} \geq 0 
   \end{aligned}
   \end{equation*}

Just as during the spatial update, we select some units (:math:`i`), and
update their calcium dynamics (:math:`\mathbf{C}_i`) based on the
**error** and the **l1-norm** of the **spikes**
(:math:`\mathbf{G}_i \cdot \mathbf{C}_i`). Again, it does not make sense
to have negative calcium dynamics or spikes, so that is a constraint on
the problem. Moreover, we need an :math:`\alpha` to provide balance
between fidelity and sparsity, which can be scaled up and down with
``sparse_penal`` (``sparse_penal=1`` is equivalent to the default
behavior of CaImAn). Furthermore, :math:`\alpha` should depend on the
expected level of noise. Note that we cannot use ``sn_spatial`` since
that was the noise for each pixel, and we need the noise for each unit.
The function ``update_temporal`` estimates the noise of each unit for
you – you just have to tell it the ``noise_freq``\ uency. Like before,
**0.5** is the highest you can go. With the default,
``noise_freq=0.25``, the higher frequency half of the signal will be
considered noise. In addition to affecting the estimation of noise
power, ``noise_freq`` affects another smoothing process: when estimating
:math:`\gamma_i`\ s, it is usually helpful to run a filter on the signal
to get rid of high freqeuency noise, particularly when you don’t have a
large ``add_lag``. The parameter, ``noise_freq`` is the cut-off
frequency of the low-pass filter run on the temporal trace for each
unit. Additionally, you can set the value of ``use_smooth`` to control
whether the filtering is done at all. Even with this careful design,
however, it is sometimes hard to approach the true solution to the
problem. When that happens, ``update_temporal`` will warn you by saying
something like “problem solved sub-optimally”. Usually, a few of these
warnings is OK, but if you see this warning a lot it either means your
parameters are unreasonable or you need more iterations to approach the
real answer. You can use ``max_iters`` to control how many iterations to
run for each small problem before the computer gives up and throws a
warning. Furthermore, in some very, very rare cases, the default `ecos
solver <https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver>`__
(the algorithm that does all the heavy-lifting) can fail and throw a
“problem infeasible” warning, and it’s worth trying a different solver,
namely
`scs <https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver>`__.
Be aware that scs produces results with very, very slow performance. The
boolean parameter ``scs_fallback`` controls whether the scs attempt
should be made before giving up. Importantly, both increasing
``max_iters`` and using ``scs_fallback`` will significantly increase the
computation time and will not help at all if the parameters you provided
are unreasonable to begin with, so try to use this only as a last
resort.

Finally, after the optimization is done, and just like
```update_spatial`` <#first-spatial-update>`__, we have a ``zero_thres``
to get rid of the small numbers, after which we can do a ``post_scal``
to counter the artifacts introduced by the **l1-norm** penalty.

``update_temporal`` takes in ``Y``, ``A``, ``b``, ``C``, ``f``, and
``sn_spatial`` (even if we won’t need it by default), in that order.
Optionally you can pass in ``noise_freq``, ``p``, ``add_lag``,
``jac_thres``, ``use_spatial``, ``sparse_penal``, ``max_iters``,
``use_smooth``, ``scs_fallback``, ``zero_thres`` and ``post_scal``, as
we have discussed. ``update_temporal`` returns much more than we
expected – in addition to ``C_temporal`` and ``S_temporal``, which are
the results we care most about, it also returns ``YrA``, and
``g_temporal`` (the :math:`\mathbf{G}` matrix for each unit). Moreover,
it returns ``B_temporal``, ``C0_temporal`` and ``sig_temporal``,
representing the final layer of complexity: when we update the temporal
trace, there might be a global baseline calcium concentration, which is
modeled by :math:`b` and returned in ``B_temporal``. A spike may also
have happened right before recording starts and the resulting calcium
transient could still be decaying in the first few seconds, so we model
this with an initial calcium concentration, :math:`c_0`, that follows
the same decaying pattern defined by :math:`\gamma_i`\ s, and is
returned in ``C0_temporal``. Both :math:`b` and :math:`c_0` are single
numbers that get updated along with the calcium dynamics for each unit.
Finally there is ``sig_temporal`` which is the combination of all the
signals, that is: ``C_temporal + C0_temporal + B_temporal``

.. container:: alert alert-success

   You should now have an idea of what each parameter is doing in
   ``update_temporal``, and be able to make sense of the visualization
   results of the parameter exploring steps.

   -  As was briefly mentioned before, minian’s output of dropped sample
      units information and visualization of their raw traces is useful
      after the first temporal update. Since one of the main purposes of
      the first temporal update is to get rid of trash cells and cells
      with noisy signal, successful parameter selection is evidenced by
      dropped units with raw traces that look like noise (no clear
      bursts of activity). Alternatively, if cell-like activity is seen
      in the raw trace of a dropped unit, this may indicate that the
      selected parameters are too conservative.

   -  When reading the temporal trace plot, “fitted spikes” (green),
      “fitted signal” (orange), and “fitted calcium trace” (blue), are
      all alligned to the “raw signal” based upon the model. Ideally, we
      want only one spike for each burst of signal, with “fitted signal”
      and “fitted calcium trace” decaying in a manner that follows the
      raw signal. Below is the temporal plot of an example unit using
      different sparse_panel:

   .. rubric:: Example Temporal Traces
      :name: example-temporal-traces

   .. figure:: img/first_tem_param.png
      :alt: example temporal traces

      example temporal traces

   Here, the top trace is when sparse_panel = 1, and we can see that
   there are lots of small spikes at the bottom, indicating we may want
   to increase the sparse_panel to get rid of them. However, when we are
   using sparse_panel = 10 (bottom panel), it’s clear that we are
   missing real spikes from raw signal. Thus, the middle panel with
   sparse_panel = 3 fits the raw signal the best here.

The code below produces plots of temporal traces and spikes after the
first temporal update and allows us to compare them to the signal
originiating from the initialization step.

.. code:: ipython3

    %%time
    YrA, C_temporal, S_temporal, B_temporal, C0_temporal, sig_temporal, g_temporal, scale = update_temporal(
        Y, A_spatial, b_spatial, C_spatial, f_spatial, sn_spatial, **param_first_temporal)
    A_temporal = A_spatial.sel(unit_id = C_temporal.coords['unit_id'])

.. code:: ipython3

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)
    (regrid(hv.Image(C_init.compute().rename('ci'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace Initial")
     + hv.Div('')
     + regrid(hv.Image(C_temporal.compute().rename('c1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace First Update")
     + regrid(hv.Image(S_temporal.compute().rename('s1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Spikes First Update")
    ).cols(2)

The following cell of code allows us to visualize units that were
dropped during the first temporal update.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        h, w = A_spatial.sizes['height'], A_spatial.sizes['width']
        im_opts = dict(aspect=w/h, frame_width=500, cmap='Viridis')
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = list(set(A_spatial.coords['unit_id'].values) - set(A_temporal.coords['unit_id'].values))
        bad_units.sort()
        if len(bad_units)>0:
            hv_res = (hv.NdLayout({
                "Spatial Footprin": regrid(hv.Dataset(A_spatial.sel(unit_id=bad_units).compute().rename('A'))
                                           .to(hv.Image, kdims=['width', 'height'])).opts(**im_opts),
                "Spatial Footprints of Accepted Units": regrid(hv.Image(A_temporal.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**im_opts)
            })
                      + datashade(hv.Dataset(YrA.sel(unit_id=bad_units).rename('raw'))
                                  .to(hv.Curve, kdims=['frame'])).opts(**cr_opts).relabel("Temporal Trace")).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

Lastly, we can visualize the activity of each unit. There are four
traces in the top plot: “Raw Signal” corresponds to ``YrA``, “Fitted
Spikes” to ``S_temporal``, “Fitted Calcium Trace” to ``C_temporal`` and
“Fitted Signal” to ``sig_temporal``. The latter two traces usually
overlap with each other since ``B_temporal`` and ``C0_temporal`` are
often equal **0**. Sadly, due to large number of frames and the
limitation of our browser, it is usually only possible to visualize 50
units at a time, hence ``select(unit_id=slice(0, 50))``. Nevertheless it
gives us an idea of how things went. Put in other numbers if you want to
see other units.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(visualize_temporal_update(YrA.compute(), C_temporal.compute(), S_temporal.compute(),
                                          g_temporal.compute(), sig_temporal.compute(), A_temporal.compute()))

merge units
-----------

One thing CNMF cannot do is merge together units that belong to the same
cell. Even though we tried something similar during
`initialization <#initialization>`__, we might miss some, and it is
better to do it here again. Recall the parameters:

.. code:: python

   param_first_merge = {
       'thres_corr': 0.9}

The idea is straight-forward and based purely on pearson correlation of
temporal activities. Any units whose spatial footprints share at least
one pixel are considered potential targets for merging, and any of these
units that have a pearson correlation of temporal activities higher than
``thres_corr`` will be merged.

.. code:: ipython3

    %%time
    A_mrg, sig_mrg, add_list = unit_merge(A_temporal, sig_temporal, [S_temporal, C_temporal], **param_first_merge)
    S_mrg, C_mrg = add_list[:]

Now you can visualize the results of unit merging. The left panel shows
the original temporal signal, while the right panel shows the temporal
signal after merging.

.. container:: alert alert-info

   Ideally, you want to see units in the left panel with too similar of
   signals, merged in the right penal. Adjust the thres_corr in
   param_first_merge accordingly.

.. code:: ipython3

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)
    (regrid(hv.Image(sig_temporal.compute().rename('c1'), kdims=['frame', 'unit_id'])).relabel("Temporal Signals Before Merge").opts(**opts_im) +
    regrid(hv.Image(sig_mrg.compute().rename('c2'), kdims=['frame', 'unit_id'])).relabel("Temporal Signals After Merge").opts(**opts_im))

test parameters for spatial update
----------------------------------

This section is almost identical to the `first
time <#test-parameters-for-first-spatial-update>`__ we explore spatial
parameters, except for changes in variable names.

.. code:: ipython3

    if interactive:
        units = np.random.choice(A_mrg.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_mrg.sel(unit_id=units).persist()
        sig_sub = sig_mrg.sel(unit_id=units).persist()

.. container:: alert alert-info

   Again, you can simply add the values that you want to test to
   sprs_ls. Pragmatically, it’s generally fine to use the same sprs_ls
   from the first spatial update or one that is a little smaller.

.. code:: ipython3

    %%time
    if interactive:
        sprs_ls = [0.001, 0.005, 0.01]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_b, cur_C, cur_f = update_spatial(
                Y, A_sub, b_init, sig_sub, f_init,
                sn_spatial, dl_wnd=param_second_spatial['dl_wnd'], sparse_penal=cur_sprs)
            if cur_A.sizes['unit_id']:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = cur_C.compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=['sparse penalty'])

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

.. container:: alert alert-info

   Again, use the visualization results here to help choose the
   sparse_panel and dl_wnd, to use in the next step. Be sure to update
   the paramaters.

second spatial update
---------------------

Below is the second iteration of the spatial update. It is identical to
`first spatial update <#first-spatial-update>`__, with the exception of
appending **it2**\ s after the variable names, standing for “iteration
2”. From this, it should be apparentt that if you you can modify the
code to have more cycles of spatial updates followed by temporal
updates. Simply add more sections like this and `the section
below <#second-temporal-update>`__.

.. code:: ipython3

    %%time
    A_spatial_it2, b_spatial_it2, C_spatial_it2, f_spatial_it2 = update_spatial(
        Y, A_mrg, b_spatial, sig_mrg, f_spatial, sn_spatial, **param_second_spatial)

.. code:: ipython3

    hv.output(size=output_size)
    opts = dict(aspect=A_spatial_it2.sizes['width']/A_spatial_it2.sizes['height'], frame_width=500, colorbar=True, cmap='Viridis')
    (regrid(hv.Image(A_mrg.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints First Update")
    + regrid(hv.Image((A_mrg.fillna(0) > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints First Update")
    + regrid(hv.Image(A_spatial_it2.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**opts).relabel("Spatial Footprints Second Update")
    + regrid(hv.Image((A_spatial_it2 > 0).sum('unit_id').compute().rename('A'), kdims=['width', 'height']), aggregator='max').opts(**opts).relabel("Binary Spatial Footprints Second Update")).cols(2)

Here again, visualize the result of second spatial update, if not
satisfying with this, feel free to reset **param_second_spatial** and
rerun this session.

.. code:: ipython3

    hv.output(size=output_size)
    opts_im = dict(aspect=b_spatial_it2.sizes['width'] / b_spatial_it2.sizes['height'], frame_width=500, colorbar=True, cmap='Viridis')
    opts_cr = dict(aspect=2, frame_height=int(500 * b_spatial_it2.sizes['height'] / b_spatial_it2.sizes['width']))
    (regrid(hv.Image(b_spatial.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial First Update')
     + datashade(hv.Curve(f_spatial.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal First Update')
     + regrid(hv.Image(b_spatial_it2.compute(), kdims=['width', 'height'])).opts(**opts_im).relabel('Background Spatial Second Update')
     + datashade(hv.Curve(f_spatial_it2.compute(), kdims=['frame'])).opts(**opts_cr).relabel('Background Temporal Second Update')
    ).cols(2)

test parameters for temporal update
-----------------------------------

This section is almost identical to the `first
time <#test-parameters-for-first-temporal-update>`__ except for variable
names.

.. code:: ipython3

    if interactive:
        units = np.random.choice(A_spatial_it2.coords['unit_id'], 10, replace=False)
        units.sort()
        A_sub = A_spatial_it2.sel(unit_id=units).persist()
        C_sub = C_spatial_it2.sel(unit_id=units).persist()

.. container:: alert alert-info

   Generally, our aim here for the second temporal update is too refine
   the model and make the “fitted spikes”, “fitted signal”, and “fitted
   calcium trace” fit the “raw signal” better.

.. code:: ipython3

    %%time
    if interactive:
        p_ls = [1]
        sprs_ls = [0.01, 0.05, 0.1]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = compute_trace(Y, A_sub, b_spatial, C_sub, f_spatial).persist()
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(p_ls, sprs_ls, add_ls, noise_ls):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print("p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}"
                  .format(cur_p, cur_sprs, cur_add, cur_noise))
            YrA, cur_C, cur_S, cur_B, cur_C0, cur_sig, cur_g, cur_scal = update_temporal(
                Y, A_sub, b_spatial, C_sub, f_spatial, sn_spatial, YrA=YrA,
                sparse_penal=cur_sprs, p=cur_p, use_spatial=False, use_smooth=True,
                add_lag = cur_add, noise_freq=cur_noise)
            YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (
                YrA.compute(), cur_C.compute(), cur_S.compute(), cur_g.compute(), cur_sig.compute(), A_sub.compute())
        hv_res = visualize_temporal_update(
            YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict,
            kdims=['p', 'sparse penalty', 'additional lag', 'noise frequency'])

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(hv_res)

second temporal update
----------------------

This section is identical to the `first temporal
update <#first-temporal-update>`__ except for variable names.

.. code:: ipython3

    %%time
    YrA, C_temporal_it2, S_temporal_it2, B_temporal_it2, C0_temporal_it2, sig_temporal_it2, g_temporal_it2, scale_temporal_it2 = update_temporal(
        Y, A_spatial_it2, b_spatial_it2, C_spatial_it2, f_spatial_it2, sn_spatial, **param_second_temporal)
    A_temporal_it2 = A_spatial_it2.sel(unit_id=C_temporal_it2.coords['unit_id'])
    g_temporal_it2 = g_temporal_it2.sel(unit_id=C_temporal_it2.coords['unit_id'])
    A_temporal_it2 = rechunk_like(A_temporal_it2, A_spatial_it2)
    g_temporal_it2 = rechunk_like(g_temporal_it2, C_temporal_it2)

.. code:: ipython3

    hv.output(size=output_size)
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap='Viridis', logz=True)
    (regrid(hv.Image(C_mrg.compute().rename('c1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Temporal Trace First Update")
     + regrid(hv.Image(S_mrg.compute().rename('s1'), kdims=['frame', 'unit_id'])).opts(**opts_im).relabel("Spikes First Update")
     + regrid(hv.Image(C_temporal_it2.compute().rename('c2').rename(unit_id='unit_id_it2'), kdims=['frame', 'unit_id_it2'])).opts(**opts_im).relabel("Temporal Trace Second Update")
     + regrid(hv.Image(S_temporal_it2.compute().rename('s2').rename(unit_id='unit_id_it2'), kdims=['frame', 'unit_id_it2'])).opts(**opts_im).relabel("Spikes Second Update")).cols(2)

Here we visualize all the units that are dropped during this step.

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        h, w = A_spatial_it2.sizes['height'], A_spatial_it2.sizes['width']
        im_opts = dict(aspect=w/h, frame_width=500, cmap='Viridis')
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = list(set(A_spatial_it2.coords['unit_id'].values) - set(A_temporal_it2.coords['unit_id'].values))
        bad_units.sort()
        if len(bad_units)>0:
            hv_res = (hv.NdLayout({
                "Spatial Footprin": regrid(hv.Dataset(A_spatial_it2.sel(unit_id=bad_units).compute().rename('A'))
                                           .to(hv.Image, kdims=['width', 'height'])).opts(**im_opts),
                "Spatial Footprints of Accepted Units": regrid(hv.Image(A_temporal_it2.sum('unit_id').compute().rename('A'), kdims=['width', 'height'])).opts(**im_opts)
            })
                      + datashade(hv.Dataset(YrA.sel(unit_id=bad_units).compute().rename('raw'))
                                  .to(hv.Curve, kdims=['frame'])).opts(**cr_opts).relabel("Temporal Trace")).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(visualize_temporal_update(YrA.compute(), C_temporal_it2.compute(), S_temporal_it2.compute(),
                                          g_temporal_it2.compute(), sig_temporal_it2.compute(), A_temporal_it2.compute()))

save results
------------

Finally, we save our results in the ``minian`` dataset. Note that you
can save any other variables by calling ``save_minian`` and using the
code below as a reference. For example, you might want to consider using
``sig_temporal`` instead of ``C_temporal`` for your subsequent analysis.
Also, you are not restricted to use the
`netcdf <https://www.unidata.ucar.edu/software/netcdf/>`__ format,
though it is recommended. `Explore the xarray
documentation <http://xarray.pydata.org/en/stable/io.html>`__ for all IO
options, and moreover, `numpy IO
capabilities <https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.io.html>`__,
since ``xarray`` is built on top of ``numpy``.

.. code:: ipython3

    %%time
    A_temporal_it2 = save_minian(A_temporal_it2.rename('A'), **param_save_minian)
    C_temporal_it2 = save_minian(C_temporal_it2.rename('C'), **param_save_minian)
    S_temporal_it2 = save_minian(S_temporal_it2.rename('S'), **param_save_minian)
    g_temporal_it2 = save_minian(g_temporal_it2.rename('g'), **param_save_minian)
    C0_temporal_it2 = save_minian(C0_temporal_it2.rename('C0'), **param_save_minian)
    B_temporal_it2 = save_minian(B_temporal_it2.rename('B'), **param_save_minian)
    b_spatial_it2 = save_minian(b_spatial_it2.rename('b'), **param_save_minian)
    f_spatial_it2 = save_minian(f_spatial_it2.rename('f'), **param_save_minian)

visualization
-------------

Here we load the data we just saved for visualization purposes.

.. code:: ipython3

    minian = open_minian(dpath,
                         fname=param_save_minian['fname'],
                         backend=param_save_minian['backend'])
    varr = load_videos(dpath, **param_load_videos)
    chk = get_optimal_chk(varr.astype(float), dim_grp=[('frame',), ('height', 'width')])
    varr = varr.chunk(dict(frame=chk['frame']))

The following cell calls ``generate_videos`` to create a video that can
help us quickly visualize the results. Under default settings, this
video will be saved in your data folder. ``generate_videos`` takes in
the dataset that contains cnmf results, an array representation of the
raw video, the full path to the output video file, and a ``dict``
specifying chunks for performance. The resulting video will have four
parts - Top left is the **Raw Video** after pre-processing and motion
correction ``minian['org']``; Top right is the **Processed Video**
``minian['Y']`` (that is, after pre-processing and motion correction);
Bottom left is the **Residule**, that is **Raw Video** - **Units**.
Bottom right is the **Units** from CNMF
``minian['A'].dot(minian['C'], 'unit_id')``;

.. code:: ipython3

    %%time
    generate_videos(
        minian, varr, dpath, param_save_minian['fname'] + ".mp4", scale='auto')

Here we have a ``CNMFViewer`` to visualize the final results.

.. container:: alert alert-info

   Top Left panel– spatial footprints of all cells (a sum projection).

   | Top Middle panel ``if UseAC`` – the dot product of A (spatial
     footprint) and C (temporal activities) matrix of selected neurons.
   | ``if not UseAC`` – the spatial footprints of selected neurons (a
     sum projection).

   Top Right panel– raw video after pre-processing and motion
   correction, which is the movie that’s fed in as ``org`` to
   ``CNMFViewer``, if nothing is fed in it’s ``minian['org']``.

.. container:: alert alert-info

   The Bottom Left Controller Panel has several useful features:

   Refresh – refreshes the data when you switch to a new group of units
   and it is not loading properly.

   Load Data – loads the data into memory, which will take some time by
   itself, but will make the later visualization faster.

   UseAC check box – choose whether or not you want the middle panel to
   be the dot product of A (spatial footprint) and C (temporal
   activities) matrix of selected neurons. Note that this will make
   visualization process slower.

   Normalize – normalizes the bottom middle trace and spike plot for
   each unit to itself.

   ShowC – shows calcium traces for each unit across time in the bottom
   middle plot.

   ShowS – shows spikes for each unit across time in the bottom middle
   plot.

   Previous Group and Next Group buttons – allow you to easily go
   backward/forward to another group of units.

   Video Play Panel – lets you play the top middle and right panel in
   real time.

.. container:: alert alert-info

   The Bottom Middle Panel contains plots of units along the time axis.
   Each group will have 4-5 units showing in the plot. Combine the plot
   with the videos to check the quality of your CNMF results.

.. container:: alert alert-info

   The Bottom Right panel. is a labeling tool for you to manually kick
   out “bad” units by labelling them (they will be demarcated with a
   ``-1``). You can also flag units to be merged if their temporal
   activities and spatial footprint suggest they should be.

.. code:: ipython3

    %%time
    if interactive:
        cnmfviewer = CNMFViewer(minian)

.. code:: ipython3

    hv.output(size=output_size)
    if interactive:
        display(cnmfviewer.show())

The following code cell serves to save your manually changed labels

.. code:: ipython3

    if interactive:
        save_minian(cnmfviewer.unit_labels, **param_save_minian)
