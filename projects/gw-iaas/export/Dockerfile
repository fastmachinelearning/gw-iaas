# start from CUDA container and install TensorRT on top
# of this using the instructions found at
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#maclearn-net-repo-install
ARG CUDA_MAJOR_VERSION=11
ARG CUDA_MINOR_VERSION=4
ARG CUDA_PATCH_VERSION=2
FROM nvcr.io/nvidia/cuda:${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.${CUDA_PATCH_VERSION}-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ARG TRT_VERSION=8.2.0
ARG CUDA_MAJOR_VERSION
ARG CUDA_MINOR_VERSION
RUN set +x \
        \
        && version="${TRT_VERSION}-1+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}" \
        \
        && apt-get update \
        \
        && apt-get install -y --no-install-recommends \
            libnvinfer8=${version} \
            libnvonnxparsers8=${version} \
            libnvparsers8=${version} \
            libnvinfer-plugin8=${version} \
            python3-libnvinfer=${version} \
        \
        && apt-mark hold \
            libnvinfer8=${version} \
            libnvonnxparsers8=${version} \
            libnvparsers8=${version} \
            libnvinfer-plugin8=${version} \
            python3-libnvinfer=${version} \
        \
        && rm -rf /var/lib/apt/lists/*


# install pip and get rid of python3 alias
RUN set +x \
        \
        && apt-get update \
        \
        && apt install -y --no-install-recommends \
            python3-pip \
            python-is-python3 \
            git \
        \
        && python -m pip install --upgrade pip \
        \
        && rm -rf /var/lib/apt/lists/*

# copy just the code needed to run the export script
ENV HOME=/home/docker
COPY libs/hermes/hermes.quiver $HOME/gw-iaas/libs/hermes/hermes.quiver
COPY libs/hermes/hermes.typeo $HOME/gw-iaas/libs/hermes/hermes.typeo
COPY projects/gw-iaas/export $HOME/gw-iaas/projects/gw-iaas/export
WORKDIR $HOME/gw-iaas/projects/gw-iaas/export

# install poetry and use it to install
# export script dependencies to global
# Python environment, not virtual env.
# TODO: Something about doing a sudo install
# using poetry causes the poetry-installed
# libs to get installed at /usr/lib/python3.8/site-packages
# instead of either /usr/lib/python3.8/dist-packages,
# which I think happens because Poetry uses pip
# under the hood, or /usr/local/lib/python3.8/dist-packages,
# which I think is caused both by pip as well
# as the sudo nature of the install. Currently "solving"
# this with a lazy copy. Should probably either update
# the PYTHONPATH environment variable or possibly
# do a symlink instead?
ARG POETRY_VERSION=1.2.0a2
RUN set -x \
        \
        && python -m pip install --upgrade poetry==${POETRY_VERSION} \
        \
        && poetry config virtualenvs.create false \
        \
        && poetry install --without dev --no-interaction \
        \
        && cp -r \
            /usr/lib/python3.8/site-packages/* \
            /usr/local/lib/python3.8/dist-packages

ENTRYPOINT ["python", "export.py"]
CMD [""]
