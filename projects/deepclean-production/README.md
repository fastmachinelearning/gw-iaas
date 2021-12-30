# DeepClean in Production
This repo contains an implementation of the DeepClean client code contained in `projects/gw-iaas/clients`, rearchitected to address a couple simplifications made for the purposes of the paper. In particular:

- This is designed for a data replay stream in which the strain and witness channels are contained in different files, written to different directories.
- The `deepcleaner.utils.FrameWriter` class has been rebuilt with postprocessing meant to address some bugs which have been reducing the quality of the clean (but this is still very much a work in progress).
    -  To this end, `end-to-end.ipynb` includes an analysis of some of the postprocessing issues of using DeepClean in an online fashion.

For executing this pipeline, I would strongly recommend using Poetry >=1.2. The instructions for how to do this aren't entirely clear on Poetry's website, so I'll give you the command you'll want to run for installation (on Linux) here:

```console
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py \
    | python - --preview
```

## Running
This experiment is expected to be run locally on a node on the LIGO Data Grid (LDG) with Singularity installed and access to a data replay stream.

### 1. Model export
First we'll need to export a DeepClean model to a local Triton model repository. Luckily we can just leverage the code used to export DeepClean for the GW-IaaS paper, detailed instructions for which I'll direct you to [the GW-IaaS README](https://github.com/fastmachinelearning/gw-iaas/tree/main/projects/gw-iaas#1-model-export). If you just want to run things with all the defaults contained in `pyproject.toml`, you can just run (with the correct `WEIGHTS_PATH` inserted):

```console
$ cd ../gw-iaas/export
$ poetry install
$ poetry run /bin/bash -c \
    "WEIGHTS_PATH=/path/to/weights.pt export-model \
        --typeo ../../deepclean-production:export:deepclean"
```

### 2. Start the server
Next we'll deploy the Triton Inference Server via a Singularity container hosted by the Open Science Grid (OSG). Note that you need to use a system with enterprise grade GPUs to run Triton (i.e. Titan or GTX GPUs won't work).

```
$ GPU_ID=... # pick a GPU ID, or multiple, to host DeepClean on
$ singularity exec --nv \
    /cvmfs/singularity.opensciencegrid.org/alec.gunny/deepclean-prod\:server-20.07 \
        /bin/bash -c \
            "CUDA_VISIBLE_DEVICES=$GPU_ID /opt/tritonserver/bin/tritonserver \
            --model-repository $HOME/repos/deepclean-online-production"
```

### 3. Run the client
In a new terminal, we'll start up a client pipeline to read from our data replay and make inference requests to the server.

#### 3a. Environment set up
Since this experiment runs client code locally, the environment set up is a bit more complicated. Specifically, the gravitational wave frame (`.gwf`) file readers in GWpy require C++ libraries that can only be installed via conda, so make sure that you have both [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) _and_ Poetry installed.

You'll then need to configure poetry to install libraries in your default Python site-packages location, instead of in a virtual environment, so that the conda packages and the ones installed by Poetry can live side-by-side. I'll use the `--local` flag for the command below, so that this config option is only set for this project (rather than globally):

```console
$ poetry config virtualenvs.create false --local
```

Now create the `gwftools` virtual environment to install all our GWpy dependencies

```console
$ conda env create -f ../../libs/hermes/hermes.gwftools/environment.yaml
```

Next we'll activate this environment and install all our Poetry-managed dependencies inside of it

```console
$ conda activate gwftools
$ poetry install
```

#### 3b. Running
This will once again look pretty similar to the matching instructions in the [the GW-IaaS README](https://github.com/fastmachinelearning/gw-iaas/tree/main/projects/gw-iaas#running-1).

If you want to run with the default values specified in `pyproject.toml`, the assumption will be that your replay stream lives in a directory `DATA_DIR` with subdirectories `lldetchar/H1` for witness channels and `llhoft/H1` for the strain channel, and that the filename patterns in these directories are identical except that `Detchar` is replaced with `HOFT` for the strain filenames. Assuming you have access to a stream like this, all you need to do is (with the correct `DATA_DIR` path inserted)

```console
$ DATA_DIR=/path/to/data deepclean --typeo :deepclean
```

You should find cleaned frames from this run in the directory `$HOME/frames/deepcleaned/aggregated-0.5s`, and logs from this run in this directory at `deepclean.latency-0.5.log`.
