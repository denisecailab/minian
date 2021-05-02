Frequently Asked Questions
==========================

Can I have a version of pipeline without all the notes?
-------------------------------------------------------

Yes! all the tutorial cells in the pipeline are tagged with "tutorial".
We can use ``nbconvert`` to obtain a copy of the notebook with `all those cells removed <https://nbconvert.readthedocs.io/en/latest/removing_cells.html#removing-pieces-of-cells-using-cell-tags>`_.
For example, the following command will give you a clean pipeline:

.. code-block:: console

    jupyter nbconvert pipeline.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="tutorial" \
    --to notebook \
    --output pipeline_clean.ipynb

Just note that since this is a different copy of the pipeline, there might be extra work if you want to merge you edits on this clean copy with updates in the default pipeline notebook.

``KilledWorker`` dammit, what do I do?
--------------------------------------

Short answer: sometimes all you need is to try again.
If that doesn't work or if that's getting annoying, then increasing ``memory_limit`` of the ``cluster`` would most likely fix it.
See :ref:`killedworker` for more detail.

What do I do with the output of MiniAn?
---------------------------------------

Hopefully the tutorials in the pipeline would give you an idea of how to interprete each variable.
Then if you're happy to work with `xarray`, then you will feel right at home once you do :py:func:`open_minian <minian.utilities.open_minian>`.
See :ref:`tips/variables:Saving to other formats` for details on converting the output to other formats.