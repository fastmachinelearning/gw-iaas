FROM python:3.9-slim-buster
ARG POETRY_VERSION=1.0.0
ENV DEBIAN_FRONTEND="noninteractive"

RUN set -ex \
        \
        && apt-get update && apt-get install -y --no-install-recommends git \
        \
        && pip install poetry=$POETRY_VERSION \
        \
        && git clone https://github.com/alecgunny/gw-iaas.git \
        \
        && cd gw-iaas/projects/slideshow && echo pwd && poetry install \
        \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /gw-iaas/projects/slideshow
ENTRYPOINT ./run.sh
