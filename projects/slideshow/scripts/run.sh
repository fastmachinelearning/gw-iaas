#!/bin/bash -e

cd "$(dirname "$0")"/..
git pull origin dev

poetry run jupyter notebook \
    --no-browser \
    --NotebookApp.token "$(get_secret jupyter-token)" \
    --ip 0.0.0.0
