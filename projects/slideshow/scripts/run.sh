#!/bin/bash -e

cd "$(dirname "$0")"
poetry install
poetry run jupyter contrib nbextension install --sys-prefix
poetry run jupyter nbextension enable splitcell/splitcell

poetry run jupyter notebook \
    --no-browser \
    --NotebookApp.token "$(get_secret jupyter-token)" \
    --ip 0.0.0.0
