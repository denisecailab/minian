Contributing to MiniAn
======================

We'd love feedback and contribution from the community!
:ref:`Fork and clone MiniAn from source <clone-source>`, make you changes and submit a PR!
Below are some book-keeping notes.

Code Style
----------

MiniAn use the `black coding style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_.

Create Release
--------------

#. Merge all branches for the release into `master`.
#. ``git checkout master``
#. ``bash scripts/create_release.sh x.x.x`` where ``x.x.x`` is version number

The script will do some checks on the Git repo, creates a new tag and updates the VERSION file with the new release number.

Packaging for PyPi
------------------

#. ``python3 setup.py sdist bdist_wheel``
#. ``pip install --upgrade twine``
#. ``python3 -m twine upload dist/*``

.. seealso:: `packaging <https://packaging.python.org/tutorials/packaging-projects/>`_

Packaging for conda-forge
-------------------------

#. fork and clone `https://github.com/conda-forge/minian-feedstock`
#. ``conda config --add channels conda-forge``
#. ``conda install conda-build``
#. ``conda install conda-smithy``
#. ``conda-build recipes/minian``
#. create a PR to upstream

Build documentation
-------------------

MiniAn use `numpy style docstring <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
It also heavily rely on auto-generated notebooks and use a custom github-action-rtd workflow.

To build documentation locally run the following commands:

.. code-block:: console

    pip install -r requirements/requirements-doc.txt
    cd docs
    make html

This however does not include the auto-generated pages for `pipeline.ipynb` and `cross-registration.ipynb`.
To include those, create a folder `docs/source/artifact`.
Then copy the notebooks (preferably with output) and the `img` folder under there.