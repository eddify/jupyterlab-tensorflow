# docker-jupyterlab-tensorflow

I've been wanting to experiment with the new [Jupyter Labs Notebooks](https://github.com/jupyterlab/jupyterlab) in my machine learning projects.

It's described as:

```
JupyterLab is the next-generation user interface for Project Jupyter. It offers all the familiar building blocks of the classic Jupyter Notebook (notebook, terminal, text editor, file browser, rich outputs, etc.) in a flexible and powerful user inteface that can be extended through third party extensions. Eventually, JupyterLab will replace the classic Jupyter Notebook after JupyterLab reaches 1.0.
```

This dockerfile builds a jupyter lab instance long with the usual goodies:
- python 3.6
- pillow
- h5py
- matplotlib
- numpy
- pandas
- scipy
- sklearn
- tensorflow 1.5.0
- nvidia cuda 9.0.176 
- nvidia cudnn 7.0.5.15

### Requirements
- ubuntu or equivalent OS
- cuda compatible gpu https://en.wikipedia.org/wiki/CUDA
- host nvidia drivers and cuda http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation
- docker https://docs.docker.com/install/
- nvidia-docker https://github.com/NVIDIA/nvidia-docker
- docker https://docs.docker.com/install/
