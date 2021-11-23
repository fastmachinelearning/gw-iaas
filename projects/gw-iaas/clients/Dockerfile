ARG CONDA_TAG=4.10.3
FROM continuumio/miniconda3:${CONDA_TAG}

# use bash as our shell so that we can
# source the conda initialization script
SHELL ["/bin/bash", "-c"]

# copy all the libs we need into our container
ENV HOME=/home/docker HERMES=libs/hermes/hermes PROJECT=projects/gw-iaas/clients
COPY ${HERMES}.gwftools $HOME/gw-iaas/${HERMES}.gwftools
COPY ${HERMES}.quiver $HOME/gw-iaas/${HERMES}.quiver
COPY ${HERMES}.stillwater $HOME/gw-iaas/${HERMES}.stillwater
COPY ${HERMES}.typeo $HOME/gw-iaas/${HERMES}.typeo
COPY ${PROJECT} $HOME/gw-iaas/${PROJECT}

# set up poetry to install virtualenvs
# in Conda's virtualenvs location so that
# we can have all our libs in one place
ARG POETRY_VERSION=1.2.0a2
ENV POETRY_VIRTUALENVS_PATH=/opt/conda/envs
ENV CONDA_INIT=$CONDA_PREFIX/etc/profile.d/conda.sh
RUN set +x \
        \
        # build and activate the conda
        # environment for gwftools
        && conda env create \
            -f $HOME/gw-iaas/libs/hermes/hermes.gwftools/environment.yaml \
        \
        && source $CONDA_INIT \
        \
        && conda activate gwftools \
        # install poetry in this environment and use
        # it to install the client libraries
        \
        && python -m pip install poetry==${POETRY_VERSION} \
        \
        && cd $HOME/gw-iaas/${PROJECT} \
        \
        && poetry install --without dev --no-interaction

WORKDIR $HOME/gw-iaas/projects/gw-iaas/clients
