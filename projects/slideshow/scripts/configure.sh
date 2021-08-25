#!/bin/bash -e

echo '''
get_secret() {
    secret=$1
    version=${2:-1}
    gcloud secrets versions access $version --secret=$secret
}
''' >> ~/.bashrc
source ~/.bashrc

get_secret github-ssh > ~/.ssh/id_ed25519
chmod 400 ~/.ssh/id_ed25519

git clone git@github.com:alecgunny/gw-iaas.git
cd gw-iaas
git checkout slideshow

cd projects/slideshow
poetry install
poetry run jupyter contrib nbextension install --sys-prefix
poetry run jupyter nbextension enable splitcell/splitcell
