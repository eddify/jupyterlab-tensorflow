# Jupyter Lab Tensorflow Docker Image

![Build Badge](https://img.shields.io/docker/automated/eddify/jupyterlab-tensorflow.svg)

I've been wanting to experiment with the new [Jupyter Labs Notebooks](https://github.com/jupyterlab/jupyterlab) in my machine learning projects.

It's described as:

```
JupyterLab is the next-generation user interface for Project Jupyter. It offers all the familiar building blocks of the classic Jupyter Notebook (notebook, terminal, text editor, file browser, rich outputs, etc.) in a flexible and powerful user inteface that can be extended through third party extensions. Eventually, JupyterLab will replace the classic Jupyter Notebook after JupyterLab reaches 1.0.
```

This dockerfile builds a jupyter lab instance with tensorflow 1.5 and cuda 9 drivers:
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
- [nvidia-docker runtime >=2](https://github.com/NVIDIA/nvidia-docker) 
- GNU/Linux x86_64 with kernel version > 3.10
- [Docker >= 1.12](https://docs.docker.com/install/)
- NVIDIA GPU with Architecture > Fermi (2.1)
- NVIDIA drivers ~= 361.93

### How to Run

```
nvidia-docker -v /host/notebook/dir:/home/jupyterlab -p 8888:8888 run eddify/jupyterlab-tensorflow:latest
```
Note: Make sure to replace ```/host/notebook/dir``` with your own host path.

If you look at the docker output, it will output a url with a token, open the url in your browser.