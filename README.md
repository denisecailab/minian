[![Build Status](https://img.shields.io/travis/DeniseCaiLab/minian/master.svg?style=flat&label=master%20build)](https://travis-ci.org/DeniseCaiLab/minian)
[![codecov](https://codecov.io/gh/DeniseCaiLab/minian/branch/master/graph/badge.svg)](https://codecov.io/gh/DeniseCaiLab/minian)
&emsp;&emsp;
[![Build Status](https://img.shields.io/travis/DeniseCaiLab/minian/dev.svg?style=flat&label=dev%20build)](https://travis-ci.org/DeniseCaiLab/minian)
[![codecov](https://codecov.io/gh/DeniseCaiLab/minian/branch/dev/graph/badge.svg)](https://codecov.io/gh/DeniseCaiLab/minian)

# About MiniAn

MiniAn is an analysis pipeline and visualization tool inspired by both [CaImAn](https://github.com/flatironinstitute/CaImAn) and [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE) package specifically for [Miniscope](http://miniscope.org/index.php/Main_Page) data.

# Quick Start Guide

## Option #2: virtual env
1. Create a python virtual env: `python3 -m venv venv`
1. Activate the virtual enviornment you created during minian installation: `source venv/bin/activate`
1. Install ffmpeg, for debian based systems: `sudo apt install ffmpeg`
1. Install required files for AV: `sudo apt install libavformat-dev libavdevice-dev`
1. Install the MiniAn package: `pip install MiniAn`
1. Install the pipeline notebooks: `minian-install-pipeline`
1. Optional install the demo movies: `minian-install-demo`
1. Fire up jupyter: `jupyter notebook` and open the notebook "pipeline_noted.ipynb"

# Quick Start Guide for build and use from source

## Option #1: conda
1. Download and install [conda](https://conda.io/projects/conda/en/latest/).
1. Make sure your conda is up to date: `conda update -n base -c default conda`
1. Download the MiniAn package: `git clone https://github.com/DeniseCaiLab/minian.git`
1. Go into MiniAn folder you just cloned: `cd minian/`
1. Create an conda environment: `conda env create -n minian -f environment.yml`
1. Activate the conda enviornment you created during minian installation: `conda activate minian`
1. Fire up jupyter: `jupyter notebook` and open the notebook "pipeline_noted.ipynb"

## Option #2: virtual env
1. Download the MiniAn package: `git clone https://github.com/DeniseCaiLab/minian.git`
1. Go into MiniAn folder you just cloned: `cd minian/`
1. Create a python virtual env: `python3 -m venv minian`
1. Activate the virtual enviornment you created during minian installation: `source minian/bin/activate`
1. Install the dependencies: `pip install -r requirements.txt`
1. Install ffmpeg, for debian based systems: `sudo apt install ffmpeg`
1. Install required files for AV: `sudo apt install libavformat-dev libavdevice-dev`
1. Fire up jupyter: `jupyter notebook` and open the notebook "pipeline_noted.ipynb"

## Option #3: docker
1. Download and install [docker](https://docs.docker.com/install/).
1. Read the notes below first then run: `docker run -p 8888:8888 -v MY_DATA_PATH:/media denisecailab/minian:latest`
1. Once you read the following, open a browser and navigate to `localhost:8888`:

> The Jupyter Notebook is running at:  
http://<server/ip>:8888  
Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

Note: this approach runs everything in a more isolated docker container, which makes customizing/extending the code harder and may affect performance. It is recommended to use this approach only when you encounter errors during installation or running demo data with the first conda option, in which case it is highly recommended to submit an [issue](https://github.com/DeniseCaiLab/minian/issues) first as well.

As a result of isolation, you have to explicitly expose resources from within the container to your machine. In our case we want to expose two things: the network port that jupyter is listening to and a file system shared across your machine and the container. The `-p 8888:8888` argument in the docker command expose port 8888 from within the container to port 8888 on your machine. If you prefer another port, for example in case another jupyter application is already running and occupies port 8888, change the number after the colon. The `-v MY_DATA_PATH:/media` argument creates a bind mount from `MY_DATA_PATH` on your machine to `/media` within the container, so that files under this two paths are synced and act as one, and you can refer to anything under `MY_DATA_PATH` by substituting that with `/media` from the notebook within the container. See [here](https://docs.docker.com/storage/bind-mounts/) for more details and change the path to suit your needs.

As a final note, everything within the container **does not persist** across sessions except those in the bind mount, which means all the modifications you made to the default notebook under `/minian` will be reset to the state in the original image. Thus it is advised to keep a working copy of minian on your local machine and mount them in the container if you are using this in production mode.

# Generate documentation

To generate documentation MiniAn uses Sphinx with the Read The Docs theme. All docstrings are in the Google docstring format see [docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

To generate the documentation please execute these commands:
```
pip install -r requirements/requirements-doc.txt
cd docs
make html
```

After the process finishes the documentation files are in the build folder

# Packaging for Conda (Forge) and PyPi

## Conda (Forge)

First add the conda-forge channel to your config: `conda config --add channels conda-forge`
Install conda build: `conda install conda-build`
To create and upload the MiniAn package to Conda Forge install Conda Smithy: `conda install -y conda-smithy`

## PyPi

Create the dist build from the MiniAn source: `python3 setup.py sdist bdist_wheel`
Install Twine: `pip install --upgrade twine`
Upload the dist to PyPi: `python3 -m twine upload dist/*`

for testing purpose you can upload to the PyPi test repo: `python3 -m twine upload --repository testpypi dist/*`
installing the test package: `python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple minian`

see also: [packaging](https://packaging.python.org/tutorials/packaging-projects/)

# License

This project is licensed under GNU GPLv3.
