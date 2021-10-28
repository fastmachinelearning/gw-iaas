ARG TAG=20.11
FROM nvcr.io/nvidia/tensorrt:${TAG}-py3

ARG POETRY_VERSION=1.0
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

ARG REPO=fastmachinelearning BRANCH=main
RUN git clone https://github.com/${REPO}/gw-iaas.git && \
        cd gw-iaas/projects/gw-iaas/export && \
        checkout ${BRANCH} && \
        poetry export -f requirements.txt --without-hashes && \
        pip install -r requirements.txt
