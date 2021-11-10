ARG CUDA_MAJOR_VERSION=11
ARG CUDA_MINOR_VERSION=4
ARG CUDA_PATCH_VERSION=2

FROM build AS base
WORKDIR /opt/gw-iaas/projects/gw-iaas/export
RUN make

# start from CUDA container and install TensorRT on top
# of this using the instructions found at
# https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#maclearn-net-repo-install
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
WORKDIR $HOME
COPY --from=base /opt/build/dist/*.whl .
COPY --from=base /opt/build/dist/deps/*.whl ./deps/
RUN set -x \
        && ls && ls deps \
        \
        && python -m pip install ./deps/*.whl \
        \
        && python -m pip install ./*.whl \
        \
        && rm -rf *.whl deps

ENTRYPOINT ["export"]
CMD [""]
