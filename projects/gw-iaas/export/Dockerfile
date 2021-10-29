ARG TAG=20.11
FROM nvcr.io/nvidia/tensorrt:${TAG}-py3

ARG POETRY_VERSION=1.1 REPO=fastmachinelearning BRANCH=main
RUN pip install poetry==${POETRY_VERSION} && \
        poetry config virtualenvs.create false && \
        git clone https://github.com/${REPO}/gw-iaas.git && \
        cd gw-iaas/projects/gw-iaas/export && \
        git checkout ${BRANCH} && \
        poetry install
        # poetry export -f requirements.txt --without-hashes && \
        # pip install -r requirements.txt
