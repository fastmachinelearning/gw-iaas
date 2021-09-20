import argparse
import inspect
import logging
import re
import sys
import time
from queue import Empty, Queue
from threading import Event
from typing import Optional, Union

import numpy as np
from google.cloud import container_v1 as container
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    Text,
    TimeRemainingColumn,
)
from torchvision.models import resnet18
from tritonclient import grpc as triton

from hermes import quiver as qv
from hermes.cloudbreak.clouds import google as cb


class ThroughputColumn(ProgressColumn):
    """Renders throughput in inferences / s"""

    def render(self, task: "Task") -> Text:
        """Show data throughput"""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:0.1f} inf/s", style="progress.data.speed")


def export(
    model_repository_bucket: str,
    model_name: str,
    max_batch_size: int = 16,
    instances_per_gpu: int = 4,
) -> qv.ModelRepository:
    """Export a ResNet model to a Google Cloud Storage model repository

    Instantiates (and randomly initializes) a ResNet 18 model
    and creates a cloud-based model repository in Google Cloud
    to which to export it.

    Args:
        model_repository_bucket:
            The name of the cloud storage bucket to which
            to export the model. Must be _globally_ unique
        model_name:
            The name to assign to the model in the repository
        max_batch_size:
            The maximum batch size which will be passed to this
            model at inference time
        instances_per_gpu:
            Number of concurrent inference execution instances
            to host per GPU
    Returns:
        The `ModelRepository` object representing the actual
        repository location
    """

    # initialize a ResNet18 model with random weights in-memory
    nn = resnet18()
    logging.info("Instantiated ResNet18 model")

    # instatiate a model repository in a cloud bucket
    model_repository_bucket = "gs://" + model_repository_bucket
    logging.info(f"Creating cloud model repository {model_repository_bucket}")
    model_repository = qv.ModelRepository(model_repository_bucket)

    # add an entry to this (now empty) repository for
    # the model that we just created
    model = model_repository.add(model_name, platform=qv.Platform.ONNX)

    # set some config parameters
    model.config.max_batch_size = max_batch_size
    model.config.add_instance_group(count=instances_per_gpu)

    # export the version of this model corresponding to this
    # particular set of (random) weights. Specify the shape
    # of the inputs to the model and the names of the outputs
    logging.info("Exporting model to repository")
    export_path = model.export_version(
        nn, input_shapes={"x": (None, 3, 224, 224)}, output_names="y"
    )

    logging.info(f"Exported model to {model_repository_bucket}/{export_path}")

    # return the repository so we can delete it when we're done
    return model_repository


def build_cluster(
    cluster_name: str,
    cluster_zone: str,
    num_nodes: int = 2,
    gpus_per_node: int = 4,
    vcpus_per_node: int = 16,
    gpu_type: str = "t4",
    credentials: Optional[str] = None,
) -> cb.Cluster:
    """Start a GKE cluster and add a GPU node pool to it

    Start a GKE cluster using the specified credentials.
    If `credentials` is left as `None`, the path to a
    service account JSON will be looked for using the
    environment variable `GOOGLE_APPLICATION_CREDENTIALS`.

    Args:
        cluster_name:
            The name to assign to the new cluster
        cluster_zone:
            The region in which to build the cluster
        num_nodes:
            The number of GPU-enabled nodes to attach to
            the cluster once it's started
        gpus_per_node:
            The number of GPUs to attach to each node
            created in the node pool
        vcpus_per_node:
            The number of VCPUs to attach to each node
            created in the node pool
        gpu_type:
            The type of GPU to use for inference
        credentials:
            Either a string a specifying a path to a user-managed
            Google Cloud service account key, or `None`, in which
            case such a string should be attached to the
            environment variable `GOOGLE_APPLICATION_CREDENTIALS`
    Returns:
        The `Cluster` object representing the new cluster
    """

    # instantiate a manager with credentials which
    # we can use to create a new cluster
    manager = cb.ClusterManager(zone=cluster_zone, credentials=credentials)

    # create a description of the cluster that
    # we want to create, starting with a vanilla
    # default node pool for cluster management
    cluster_config = container.Cluster(
        name=cluster_name,
        node_pools=[
            container.NodePool(
                name="default-pool",
                initial_node_count=1,
                config=container.NodeConfig(),
            )
        ],
    )

    # create the cluster using the manager and
    # then wait for it to be ready
    cluster = manager.add(cluster_config)
    cluster.wait_for_ready()

    # once it's ready, deploy a daemon set which
    # installs and exposes the GPU drivers to
    # containers on each node
    cluster.deploy_gpu_drivers()

    # describe a GPU-enabled set of nodes
    # to attach to the cluster for inference
    node_pool_config = container.NodePool(
        name=f"tritonserver-{gpu_type}-pool",
        initial_node_count=num_nodes,
        config=cb.create_gpu_node_pool_config(
            vcpus=vcpus_per_node, gpus=gpus_per_node, gpu_type=gpu_type
        ),
    )

    # use the cluster to create this node pool and
    # wait for it to become ready to use
    node_pool = cluster.add(node_pool_config)
    node_pool.wait_for_ready()

    # return the cluster for deletion later
    return cluster


def do_inference(
    dataset: np.ndarray,
    server_url: str,
    model_name: str,
    model_version: int = 1,
    batch_size: int = 16,
    request_rate: float = 50,
) -> np.ndarray:
    """Perform asynchronous inference on the provided dataset

    Use an inference service hosted at `server_url` to perform
    inference on some data in batches.

    Args:
        dataset:
            The data on which to perform inference, with the
            0th axis representing individual samples
        server_url:
            The URL at which the inference service is
            awaiting requests
        model_name:
            The name of the model on the inference service
            to load and use for inference
        model_version:
            Which version of the specified model to use for inference
        batch_size:
            The number of samples over which to execute inference
        request_rate:
            How quickly to send requests to the server. Tune to your
            network speed
    Returns:
        An array representing the inference outputs for
        each sample in the dataset.
    """

    # instantiate a client that points to the appropriate address
    client = triton.InferenceServerClient(server_url)
    if not client.is_server_live():
        raise RuntimeError("Server isn't ready!")

    # manually load in the model, this way we can do things
    # like dynamically scale the amount of parallelism
    client.load_model(model_name)
    if not client.is_model_ready(model_name):
        raise RuntimeError("Model isn't ready!")

    # infer information about the model inputs
    # by querying the server for metadata
    metadata = client.get_model_metadata(model_name)
    input = metadata.inputs[0]
    input_shape = [i if i != -1 else batch_size for i in input.shape]
    input = triton.InferInput(input.name, input_shape, input.datatype)

    # set things up to use threading for asynchronous
    # request generation and handling. Include a progress
    # bar to keep track of how long things are taking
    q, e = Queue(), Event()
    progbar = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        ThroughputColumn(),
        TimeRemainingColumn(),
    )

    N = len(dataset)
    submit_task_id = progbar.add_task("Submitting requests", total=N)
    infer_task_id = progbar.add_task("Collecting results", total=N)

    # asynchronous requests require a callback function
    # that handles the response from the server in a
    # separate thread.
    def callback(result, error=None):
        """Callback function for returning responses to main thread"""

        # first check to see if there was an error
        if error is not None:
            # notify the main thread and set the stop event
            # so that we stop sending requests
            exc = RuntimeError("Encountered error in callback: " + str(error))
            q.put(exc)
            e.set()
        elif not e.is_set():
            # otherwise if we haven't hit an error yet,
            # keep parsing results
            y = result.as_numpy("y")
            progbar.update(infer_task_id, advance=len(y))

            # get the request id so we know where
            # to put this output in the array
            request_id = int(result.get_response().id)

            # send the parsed response back to the main
            # thread through the queue
            q.put((y, request_id))

    # now here is where we actually do the inference
    with progbar:
        num_batches = (N - 1) // batch_size + 1
        for i in range(num_batches):
            if e.is_set():
                # stop sending data if we hit an error
                break

            # iterate through the dataset in batches and
            # set the input message's data to the new batch
            X = dataset[i * batch_size : (i + 1) * batch_size]
            if len(X) < batch_size:
                # if the last batch didn't have enough samples,
                # pad X with some zeros and we'll slice them
                # off in the callback
                pad = np.zeros_like(X)[: batch_size - len(X)]
                X = np.concatenate([X, pad])
            input.set_data_from_numpy(X)

            # submit an asynchronous inference request
            client.async_infer(
                model_name,
                model_version=str(model_version),
                inputs=[input],
                callback=callback,
                request_id=str(i),
            )

            # update one of the tasks on our progress
            # bar to indicate how many requests we've made
            progbar.update(submit_task_id, advance=len(X))

            # do a lazy throttle on our request rate by sleeping
            time.sleep(1 / request_rate)

        # instantiate a results array and populate
        # it with data returned from the queue
        results = np.zeros((N, metadata.outputs[0].shape[1]))
        n = 0
        while n < num_batches:
            try:
                y = q.get(timeout=0.01)
                n += 1
            except Empty:
                continue

            # check to make sure an exception didn't get raised
            if isinstance(y, Exception):
                raise y
            else:
                y, request_id = y

            # add our results to our running output
            # make sure to check to see if we need
            # to clip the end off of `y`
            start = request_id * batch_size
            stop = min(start + batch_size, N)
            results[start:stop] = y[: stop - start]

    # concatenate all the responses and return them
    return results


def main(
    model_name: str,
    model_repository_bucket: str,
    cluster_name: str,
    cluster_zone: str,
    num_nodes: int = 1,
    gpus_per_node: int = 4,
    vcpus_per_node: int = 16,
    instances_per_gpu: int = 4,
    gpu_type: str = "t4",
    batch_size: int = 4,
    num_samples: int = 5000,
    request_rate: float = 50,
    credentials: Optional[str] = None,
) -> None:
    """Export a model to an inference service then do inference with it

    Instantiates and exports a ResNet 18 model to a cloud-based
    model repository, starts up an inference service on Kubernetes
    that points to it, then performs inference by sending requests
    consisting of random data to the service from your local device.
    At the end, the cloud storage bucket used to host the repository
    and the cluster used to perform inference are deleted to ensure
    that your project stops incurring costs once the script is done.

    Args:
        model_name:
            The name to give to the created model
        model_repository_bucket:
            The name to give to the cloud storage bucket used to
            host the model repository. Must be *globablly* unique
        cluster_name:
            The name to assign to the cluster used to host the
            inference service. Must be unique within the GCP
            project and zone used
        cluster_zone:
            The GCP zone to use for inference. Must have the
            request type of GPUs available. For a reference on
            which GPUs are available in what zones, see the
            support matrix [here]
            (https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)
        num_nodes:
            The number of GPU-enable nodes to use for inference
        gpus_per_node:
            The number of GPUs to assign to each node used for inference.
            The maximum allowable value will depend on the type of
            GPU used.
        vcpus_per_node:
            The number of VCPUs to assign to each node used for inference.
            The minimum allowable value will depend on the value
            of `gpus_per_node`
        instances_per_gpu:
            The number of concurent execution instances of
            the model to host per GPU
        gpu_type:
            The type of GPU to use for inference. Should match the
            last part of the strings listed [here]
            (https://cloud.google.com/compute/docs/gpus#introduction).
            E.g. `"t4"`, `"v100"`, etc.
        batch_size:
            The number of dataset samples to include with each
            inference request.
        num_samples:
            The number of inferences to perform
        request_rate:
            How quickly to send requests to the server. Tune to your
            network speed
        credentials:
            The path to a JSON file containing user-managed
            service account credentials or `None`, in which case
            such a path should be contained in the environment
            variable `GOOGLE_APPLICATION_CREDENTIALS`
    Returns:
        The model inference outputs on the random data
    """

    # instantiate a model and a model repository, then
    # export the model to that repository
    repo = export(
        model_repository_bucket=model_repository_bucket,
        model_name=model_name,
        max_batch_size=batch_size,
        instances_per_gpu=instances_per_gpu,
    )

    # create a cluster and attach some GPU-enabled nodes to it
    cluster = build_cluster(
        cluster_name=cluster_name,
        cluster_zone=cluster_zone,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        vcpus_per_node=vcpus_per_node,
        gpu_type=gpu_type,
        credentials=credentials,
    )

    # do everything else in a try-catch that way
    # if anything goes wrong we make sure to clean
    # up our resource usage
    try:
        # deploy the workload described by `tritonserver.yaml`
        # to the cluster, specifying some of the parameters
        # as variables dynamically. This workload just runs
        # the triton server on the specified number of nodes
        # and exposes them to external requests via a LoadBalancer
        # service. See the YAML for more info
        deployment, load_balancer = cluster.deploy(
            "tritonserver.yaml",
            name="tritonserver",
            replicas=num_nodes,
            gpus=gpus_per_node,
            gpuType="t4",
            modelRepo=model_repository_bucket,
            tag="20.11",
        )

        # wait for all the inference service instances to
        # come online before continuing. Use the IP assigned
        # to the load balancer to make requests
        deployment.wait_for_ready()
        server_url = f"{load_balancer.ip}:8001"

        # instantiate a random dataset in-memory. Your real
        # dataset may require more exotic schemes to be able
        # to meet the throughput capacity of the server
        dataset = np.random.randn(num_samples, 3, 224, 224).astype("float32")

        # do some inference on this dataset using the model
        # we created earlier, which is being hosted by the
        # inference service at the specified URL
        results = do_inference(
            dataset=dataset,
            server_url=server_url,
            model_name=model_name,
            model_version=1,
            batch_size=batch_size,
            request_rate=request_rate,
        )
    finally:
        # no matter what happened, clean up all of the
        # resources we were using to avoid incurring extra costs
        cluster.remove()

        logging.info(
            f"Removing cloud model repository {model_repository_bucket}"
        )
        repo.delete()

    # return the inference results for downstream processing
    return results


if __name__ == "__main__":
    # build a command line parser programatically from
    # the documentation and annotations on `main`
    doc, args = main.__doc__.split("Args:\n")
    args, _ = args.split("Returns:\n")

    parser = argparse.ArgumentParser(
        prog="Hermes Hello World",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for name, arg in inspect.signature(main).parameters.items():
        # first check if the argument has an `Optional` annotation
        try:
            if arg.annotation.__origin__ == Union:
                # if so, use the argument of the annotation
                # as the type for the argument
                type_ = arg.annotation.__args__[0]
        except AttributeError:
            # otherwise just use the annotation itself as the type
            type_ = arg.annotation

        # do a regex search to get the help description for the argument
        doc_re = "(?<={}:\n).+(?=\n{})"
        try:
            postfix = " " * 8 + "[a-z]"
            doc_str = re.search(doc_re.format(name, postfix), args).group(0)
        except AttributeError:
            # catch for the last argument, which won't
            # have another argument after it
            doc_str = re.search(doc_re.format(name, " " * 4), args).group(0)
        doc_str = " ".join([i.strip() for i in doc_str.split("\n")])

        # use dashes instead of underscores in arg names
        # then add the argument, using the default if
        # there is one, otherwise make it required
        arg_name = "--" + name.replace("_", "-")
        if arg.default == inspect._empty:
            parser.add_argument(
                arg_name, type=type_, required=True, help=doc_str
            )
        else:
            parser.add_argument(
                arg_name, type=type_, default=arg.default, help=doc_str
            )

    # parse the arguments and run `main` with them
    flags = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d - %(levelname)-8s %(message)s",
        stream=sys.stdout,
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    main(**vars(flags))
