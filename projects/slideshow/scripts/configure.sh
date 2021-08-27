#!/bin/bash -e

echo '''
# shortcut function for accessing gcloud secrets
get_secret() {
    secret=$1
    version=${2:-1}
    gcloud secrets versions access $version --secret=$secret
}

export -f get_secret
''' >> ~/.bashrc
source ~/.bashrc

# add github ssh key for development
get_secret github-ssh > ~/.ssh/id_ed25519
chmod 400 ~/.ssh/id_ed25519

# clone the repo and checkout the slideshow branch
git clone git@github.com:alecgunny/gw-iaas.git
cd gw-iaas
git checkout slideshow

# move to this project and set everything up
cd projects/slideshow
poetry install
poetry run jupyter contrib nbextension install --sys-prefix
poetry run jupyter nbextension enable splitcell/splitcell
