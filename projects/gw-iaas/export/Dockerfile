ARG TAG=21.10
FROM nvcr.io/nvidia/tensorrt:${TAG}-py3

ARG POETRY_VERSION=1.2.0a2 REPO=fastmachinelearning BRANCH=main
RUN python -m pip install --upgrade poetry==${POETRY_VERSION} \
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

WORKDIR /workspace/gw-iaas/projects/gw-iaas/export
ENTRYPOINT ["python", "export.py"]
CMD [""]
