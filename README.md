# About MiniAn

MiniAn is an analysis pipeline and visualization tool built around [CaImAn](https://github.com/flatironinstitute/CaImAn) package specifically for [Miniscope](http://miniscope.org/index.php/Main_Page) data.

# Quick Start Guide
1. Follow the [installation guide](https://github.com/flatironinstitute/CaImAn#installation-for-calcium-imaging-data-analysis) of CaImAn package to setup conda enviornment.
2. Download the MiniAn package: `git clone https://github.com/phildong/minian`
3. Activate the conda enviornment you created during CaImAn installation: `source activate caiman`
4. Go into MiniAn folder: `cd minian/`
5. [Update the enviornment](https://conda.io/docs/commands/env/conda-env-update.html) with additional packages required by MiniAn: `conda env update -f environment.yml`
6. Fire up jupyter and open the notebook "pipeline_noted.ipynb"
