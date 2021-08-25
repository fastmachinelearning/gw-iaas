#!/bin/bash -e

# install NVIDIA drivers
apt-get install -y software-properties-common
apt-get update
apt-get install linux-headers-$(uname -r)
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /"
add-apt-repository contrib
apt update
DEBIAN_FRONTEND="noninteractive" apt-get install -y cuda
export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}

# install git and python dependencies
sudo apt-get install -y \
    git \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    curl \
    libbz2-dev \
    wget \
    python3-distutils \
    python3-apt

# install python
wget https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
tar -xf Python-3.9.1.tgz
cd Python-3.9.1
./configure --enable-optimizations
make -j 2
sudo make altinstall

# install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.9 -
