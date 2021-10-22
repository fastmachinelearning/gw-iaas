# Projects
Here are various pipelines/examples built on top of the HERMES libraries for doing real-time inference-as-a-service (IaaS).

## `gw-iaas`
Contains the pipelines used to recreate the experiments outlined in [Hardware-accelerated Inference for Real-Time Gravitational-Wave Astronomy](https://arxiv.org/abs/2108.12430).

## `hello-world`
A simple, non-streaming inference-as-a-service pipline to outline some of the key concepts used in moving inference to a dedicated service.

## `slideshow`
An Jupyter Notebook RISE slideshow motivating the IaaS model and outlining some of the results covered in [Hardware-accelerated Inference for Real-Time Gravitational-Wave Astronomy](https://arxiv.org/abs/2108.12430).

## `snapshotter-test`
A streaming version of `hello-world` used to verify the behavior of the streaming snapshotter model. Will eventually be replaced by a proper integration test in the `hermes.stillwater` module.
