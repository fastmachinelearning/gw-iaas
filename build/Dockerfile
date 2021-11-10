ARG PYTHON_TAG=3.8-slim-bullseye
FROM python:${PYTHON_TAG}
ARG POETRY_VERSION=1.2.0a2
RUN set +x \
        \
        && apt-get update \
        \
        && apt-get install -y --no-install-recommends build-essential \
        \
        && python -m pip install poetry==${POETRY_VERSION} \
        \
        && rm -rf /var/lib/apt/lists/*

COPY . /opt/gw-iaas
COPY build/Makefile /opt/build/Packages.mk
ENV BUILD_DIR=/opt/build/dist
