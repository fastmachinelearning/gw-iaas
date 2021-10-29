ARG TAG=20.11
FROM nvcr.io/nvidia/tensorrt:${TAG}-py3

ARG POETRY_VERSION=1.1 REPO=fastmachinelearning BRANCH=main
RUN python -m pip install --upgrade \
            poetry==${POETRY_VERSION} \
            pip \
            setuptools \
            wheel \
        \
        && git clone https://github.com/${REPO}/gw-iaas.git \
        \
        && cd gw-iaas/projects/gw-iaas/export \
        \
        && git checkout ${BRANCH} \
        \
        && poetry config virtualenvs.create false \
        \
        && poetry install --no-dev --no-interaction \
        \
        && cp -r \
            /usr/lib/python3.8/site-packages/* \
            /usr/local/lib/python3.8/dist-packages
