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
        # && poetry export \
        #     -f requirements.txt \
        #     --output requirements.txt \
        #     --without-hashes \
        # \
        # && cat requirements.txt \
        # && python -m pip install -r requirements.txt
        && poetry config virtualenvs.create false \
        \
        && poetry install --no-dev --no-interaction
