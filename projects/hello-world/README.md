# Hermes Hello World
A brief example to illustrate the usage of Hermes tools for constructing arbitrary inference-as-a-service pipelines. Creates a cloud-based model repository then starts an inference service on Kubernetes that reads from it and sends it requests.

## How to run
Install [Poetry](https://python-poetry.org/docs/).

In order to utilize the Google Cloud Storage model repository and Google Kubernetes Engine, you'll need to create a [user-managed service account key](https://cloud.google.com/iam/docs/service-accounts#user-managed_keys) for the Google Cloud project you plan on using.

Once you have these prerequisites met, you can install the example with

```bash
poetry install
```

then `cd` into the `hello-world` directory and run the example script with

```bash
cd hello-world
poetry run python -m main -h
```

which will list the available command line options. Note that the JSON file containing the service account credentials you created as a prerequisite can be specified using the `--credentials` argument. Alternatively, you can run the script in a virtual environment by running

```bash
poetry shell
```

then set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to this path before running the script in the virtual environment

```bash
python -m main ...
```
