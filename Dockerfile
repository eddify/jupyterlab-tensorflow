# adapted from
# https://gitlab.com/nvidia/cuda/blob/ubuntau16.04/9.1/base/Dockerfile
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.1/runtime/Dockerfile
# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.1/runtime/cudnn7/Dockerfile
# https://github.com/tensorflow/tensorflow/blob/b43d0f3c98140edfebb8295ea4a4b661e2fc2a85/tensorflow/tools/docker/Dockerfile.gpu
# https://github.com/conda/conda-docker/blob/master/miniconda3/debian/Dockerfile

FROM ubuntu:16.04
MAINTAINER Eddy Kim <eddykim87@gmail.com>
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ENV CUDA_VERSION=9.0.176 \
    CUDA_PKG_VERSION="9-0=9.0.176-1" \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=9.0" \
    NCCL_VERSION=2.1.2 \
    CUDNN_VERSION=7.0.5.15

LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && apt-get install -y --no-install-recommends cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.0 /usr/local/cuda && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda9.0 libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && conda create -n jupyterlab python=3 -y \
    && conda install -c condaforge jupyterlab -n jupyterlab \
    && source activate jupyterlab \
    && pip --no-cache-dir install \ 
        Pillow \
        h5py \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes \
    && useradd -m jupyterlab

ENV PATH /opt/conda/bin:$PATH
WORKDIR /home/jupyterlab
CMD /bin/su -s \
    /bin/bash -c \
    "source activate jupyterlab && jupyter lab --ip 0.0.0.0 --port 8888 --notebook-dir /home/jupyterlab" \
    jupyterlab

EXPOSE 8888 