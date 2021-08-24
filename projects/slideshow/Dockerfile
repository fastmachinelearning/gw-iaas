FROM python:3.9-slim-buster
ARG POETRY_VERSION=1.1.7
ENV DEBIAN_FRONTEND="noninteractive"

RUN set -ex \
        \
        && apt-get update && apt-get install -y --no-install-recommends git \
        \
        && pip install poetry==$POETRY_VERSION \
        \
        && git clone https://github.com/alecgunny/gw-iaas.git \
        \
        && cd gw-iaas && pwd && git checkout slideshow \
        \
        && cd projects/slideshow && poetry install \
        \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /gw-iaas/projects/slideshow
ENTRYPOINT ./run.sh
