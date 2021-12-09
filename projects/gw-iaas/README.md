# GW-IaaS
This contains all the code for replicating the experiments outlined in [Hardware-accelerated Inference for Real-Time Gravitational-Wave Astronomy](https://arxiv.org/abs/2108.12430).

For all of these experiments, I would strongly recommend using Poetry >=1.2. Since the instructions for how to do this aren't entirely clear on Poetry's website, the command you'll want to run for installation (on Linux) is

```console
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python - --preview
```

## Sub-projects
### `clients`
Contains the client scripts for running both DeepClean and the end-to-end ensemble pipelines. Doesn't distinguish between online and offline because the difference between these use cases as far as the clients are concerned is primarily _where_ these client scripts get run (which amount to different command line argument, but not fundamentally different code).

### `export`
Contains a utility script for exporting the models used in both the DeepClean and end-to-end ensemble experiments (both online and offline).

### `offline-orchestration`
Contains the code for spinning up and tearing down cloud-based clusters and VMs for performing the offline experiments, calling the scripts from `clients` on remote VMs. Still under construction.


## Running the experiments
### DeepClean Online
This experiment is expected to be run locally on a node on the LIGO Data Grid (LDG) with Singularity installed and access to a data replay stream. Note that the channels listed in `channels.deepclean.txt` aren't the actual channels used to train DeepClean, but are used out of convenience due to their availability in the current data replay stream. As a result, the outputs of this pipeline won't have physical significance, but the infrastructure should be fully general to actually meaningful channels as well.

#### 1. Model export
First we'll need to export a DeepClean model to a local Triton model repository.
##### Environment set up
This step relies only on [Poetry](https://python-poetry.org/docs/#installation).

```console
$ cd export
$ poetry install
```

##### Running
You can see the various command line options by running (from the `export` directory):

```console
$ poetry run export-model deepclean -h
```

If you'd like to use the default values contained in `pyproject.toml`, you can just run things with (again from the `export` directory, with `/path/to/weights.pt` replaced with an actual paths to trained DeepClean weights):

```console
$ poetry run /bin/bash -c "WEIGHTS_PATH=/path/to/weights.pt export --typeo ..:export.online:deepclean"
```

You should now see 4 models in `$HOME/repos/deepclean-online`:

```console
$ ls $HOME/repos/deepclean-online
aggregator
deepclean
deepclean-stream
snapshotter
```

#### 2. Start the server
Next we'll deploy the Triton Inference Server via a Singularity container hosted by the Open Science Grid (OSG).

```
$ GPU_ID= # pick a GPU ID to host DeepClean on
$ singularity exec --nv \
    /cvmfs/singularity.opensciencegrid.org/alec.gunny/deepclean-prod\:server-20.07 \
        /bin/bash -c \
        "CUDA_VISIBLE_DEVICES=$GPU_ID /opt/tritonserver/bin/tritonserver --model-repository $HOME/repos/deepclean-online"
```

#### 3. Run the client
In a new terminal, we'll start up a client pipeline to read from our data replay and make inference requests to the server.

##### Environment set up
Since this experiment runs client code locally, the environment set up is a bit more complicated. Specifically, the gravitational wave frame (`.gwf`) file readers in GWpy require C++ libraries that can only be installed via conda, so make sure that you have both [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) _and_ Poetry installed.

You'll then need to configure poetry to install libraries in your default Python site-packages location, instead of in a virtual environment, so that the conda packages and the ones installed by Poetry can live side-by-side. This can be achieved via

```console
$ poetry config virtualenvs.create false
```

If you think you might like to use Poetry besides just within conda virtualenvs, you can activate it for just this project by

```console
$ cd clients
$ poetry config virtualenvs.create false --local
```

Now create the `gwftools` virtual environment to install all our GWpy dependencies (assuming you're running from this directory, otherwise provide the correct mapping to `libs/`)

```console
$ conda env create -f ../../libs/hermes/hermes.gwftools/environment.yaml
```

Next we'll activate this environment and install all our Poetry-managed dependencies inside of it

```console
$ cd clients
$ conda activate gwftools
$ poetry install
```

##### Running
All the following commands are expected to be run from the `clients` directory with the `gwftools` conda environment activated. As with the export step, you can see the available command line options via

```console
$ poetry run deepclean -h
```

To run with the default values specified in `pyproject.toml`, you can again run (with `/path/to/data` replaced with the actual path to your data replay stream)

```console
$ poetry run /bin/bash -c "DATA_DIR=/path/to/data deepclean --typeo ..:client.deepclean.online"
```

The cleaned frames will be available at `expts/stride-8_rate-750_instances-6_gpus-1/cleaned`, and measurements of per-model throughput and latency can be analyzed at `expts/stride-8_rate-750_instances-6_gpus-1/results.csv`.
