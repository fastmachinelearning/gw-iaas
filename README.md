# GW-IAAS

Repository for the code outlined in [Hardware-accelerated Inference for Real-Time Gravitational-Wave Astronomy](https://arxiv.org/abs/2108.12430).

## Structure
`libs` contains the various libraries used to construct inference-as-a-service pipelines.

`projects` contains pipelines built using the libraries in `libs`, including the piplines used in the paper (`offline` and `online`).

## Prerequisites
Built on top of [Poetry](https://python-poetry.org), which is required to run. Additional steps will be required for each of the various pipelines. Libraries in `libs` will be available on PyPi sometime in the near future.
