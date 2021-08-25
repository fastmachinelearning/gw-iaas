#!/bin/bash -e

get_secret() {
    secret=$1
    version=${2:-1}
    gcloud secrets versions access $version --secret=$secret
}

get_secret github-ssh > ~/.ssh/id_ed25519
chmod 400 ~/.ssh/id_ed25519

git clone git@github.com:alecgunny/gw-iaas.git
cd gw-iaas
git checkout slideshow
cd projects/slideshow
poetry install

# jupyter contrib nbextension install --sys-prefix
# jupyter nbextension enable splitcell/splitcell

poetry run jupyter notebook \
    --no-browser \
    --NotebookApp.token "$(get_secret jupyter-token)" \
    --ip 0.0.0.0
