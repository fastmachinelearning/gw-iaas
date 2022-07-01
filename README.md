# GW-IAAS
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5567703.svg)](https://doi.org/10.5281/zenodo.5567703)

_NOTE FOR HERMES USERS: HERMES DEVELOPMENT HAS PERMANENTLY MOVED TO [THIS REPOSITORY](https://github.com/ML4GW/hermes)._

WIP repository for the code used to implement the experiments in [Hardware-accelerated Inference for Real-Time Gravitational-Wave Astronomy](https://arxiv.org/abs/2108.12430). Original code used for the paper is being restructured into a more organized and general-purpose set of tools which will be made available here.

## Structure
`libs` contains the various libraries used to construct inference-as-a-service pipelines.

`projects` contains pipelines built using the libraries in `libs`, as well as simpler examples for usage and a Jupyter Notebook slideshow covering the work.

## Prerequisites
Built on top of [Poetry](https://python-poetry.org), which is required to run. Most pipelines will also require a user-managed Google Cloud [service account key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating_service_account_keys) for deploying clients and services on cloud resources.

Additional steps will be required for each of the various pipelines, please consult their individual READMEs. Libraries in `libs` will be available on PyPi sometime in the near future.
