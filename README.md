# About MiniAn

MiniAn is an analysis pipeline and visualization tool inspired by both [CaImAn](https://github.com/flatironinstitute/CaImAn) and [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE) package specifically for [Miniscope](http://miniscope.org/index.php/Main_Page) data.

# Quick Start Guide
1. Download and install [conda](https://conda.io/projects/conda/en/latest/).
2. Make sure your conda is up to date: `conda update -n base -c default conda`
2. Download the MiniAn package: `git clone https://github.com/phildong/minian.git`
3. Go into MiniAn folder you just cloned: `cd minian/`
4. Create an conda environment: `conda env create -n minian -f environment.yml`
5. Activate the conda enviornment you created during CaImAn installation: `source activate minian`
6. Fire up jupyter: `jupyter notebook` and open the notebook "pipeline_noted.ipynb"

# Additional steps for Windows:
The package `cvxpy` in `conda-forge` channel is out-dated and you would need to install with `pip`. Here are the steps:
1. Download and install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Within the `minian` environment, do: `pip install --upgrade cvxpy`
