# GW-IaaS
This contains all the code for replicating the experiments outlined in [Hardware-accelerated Inference for Real-Time Gravitational-Wave Astronomy](https://arxiv.org/abs/2108.12430). Still under construction.

## `clients`
Contains the client scripts for running both DeepClean and the end-to-end ensemble pipelines. Doesn't distinguish between online and offline because the difference between these use cases as far as the clients are concerned is primarily _where_ these client scripts get run (which amount to different command line argument, but not fundamentally different code).

## `export`
Contains a utility script for exporting the models used in both the DeepClean and end-to-end ensemble experiments (both online and offline).
