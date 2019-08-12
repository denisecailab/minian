FROM continuumio/miniconda3:latest
ADD ./minian /minian/minian
ADD ./examples /minian/examples
ADD ./demo_movies/msCam1.avi /minian/demo_movies/msCam1.avi
ADD ./img /minian/img
ADD ./pipeline.ipynb /minian/pipeline.ipynb
ADD ./pipeline_noted.ipynb /minian/pipeline_noted.ipynb
ADD ./cross-registration.ipynb /minian/cross-registration.ipynb
ADD ./environment.yml /minian/environment.yml
WORKDIR "/minian"
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN conda update -n base conda -y
RUN conda env create -n minian -f environment.yml
RUN /bin/bash -c "source activate minian && conda install -c conda-forge -y jupyterlab && jupyter labextension install @pyviz/jupyterlab_pyviz"
EXPOSE 8888
ENV PATH /opt/conda/envs/minian/bin:$PATH
ENV CONDA_DEFAULT_ENV minian
ENV CONDA_PREFIX /opt/conda/envs/minian
CMD ["jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]