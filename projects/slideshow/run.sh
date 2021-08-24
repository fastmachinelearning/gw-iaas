#!/bin/bash -e

get_secret() {
    secret=$1
    version=${2:-1}
    gcloud secrets versions access $version --secret=$secret
}

get_secret github-ssh > ~/.ssh/id_ed25519
chmod 400 ~/.ssh/id_ed25519
git remote set-url origin git@github.com:alecgunny/gw-iaas.git

TOKEN=$(get_secret jupyter-token)
poetry run jupyter notebook presentation.ipynb \
    --no-broswer \
    --NotebookApp.token $TOKEN \
    --ip 0.0.0.0
