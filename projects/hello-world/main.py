import argparse
import logging
import time
from queue import Queue
from threading import Event
from typing import Optional

import numpy as np
from google.cloud import container_v1 as container
from rich.progress import Progress
from torchvision.models import resnet18
from tritonclient import grpc as triton

from gravswell import quiver as qv
from gravswell.cloudbreak.clouds import google as cb


def export(
    model_repository_bucket: str, model_name: str, max_batch_size: int = 16
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
    Returns:
        The `ModelRepository` object representing the actual
        repository location
    """

    # initialize a ResNet18 model with random weights in-memory
    nn = resnet18()

    # instatiate a model repository in a cloud bucket
    model_repository = qv.ModelRepository("gs://" + model_repository_bucket)

    # add an entry to this (now empty) repository for
    # the model that we just created
    model = model_repository.add(model_name, platform=qv.Platform.ONNX)

    # set some config parameters
    model.config.max_batch_size = max_batch_size
    model.config.add_instance_group(count=4)

    # export the version of this model corresponding to this
    # particular set of (random) weights. Specify the shape
    # of the inputs to the model and the names of the outputs
    export_path = model.export_version(
        nn, input_shapes={"x": (None, 3, 224, 224)}, output_names="y"
    )

    logging.info(
        "Exported model to {}/{}".format(model_repository_bucket, export_path)
    )

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
    manager = cb.ClusterManager(zone=cluster_zone, credentials=credentials)

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
    cluster = manager.add(cluster_config)
    cluster.wait_for_ready()
    cluster.deploy_gpu_drivers()

    node_pool_config = container.NodePool(
        name=f"tritonserver-{gpu_type}-pool",
        initial_node_count=num_nodes,
        config=cb.create_gpu_node_pool_config(
            vcpus=vcpus_per_node, gpus=gpus_per_node, gpu_type=gpu_type
        ),
    )

    node_pool = cluster.add(node_pool_config)
    node_pool.wait_for_ready()
    cluster = manager.resources[cluster_name]

    return cluster


def do_inference(
    dataset: np.ndarray,
    server_url: str,
    model_name: str,
    model_version: int = 1,
    batch_size: int = 16,
) -> np.ndarray:
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
    progbar = Progress()

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
            q.put(error)
            e.set()

        # otherwise parse the output from the response
        # and update our progress bar
        y = result.as_numpy("y")
        progbar.update(infer_task_id, advance=len(y))

        # send the parsed response back to the main
        # thread through the queue
        q.put(y)

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
            input.set_data_from_numpy(X)

            # submit an asynchronous inference request
            client.async_infer(
                model_name,
                model_version=str(model_version),
                inputs=[input],
                callback=callback,
            )

            # update one of the tasks on our progress
            # bar to indicate how many requests we've made
            progbar.update(submit_task_id, advance=len(X))
            time.sleep(0.1)

        # now grab all the parsed responses from our output queue
        results = []
        while len(results) < len(dataset):
            y = q.get_nowait()
            if isinstance(y, Exception):
                raise y
            results.extend(y)

    # concatenate all the responses and return them
    return np.stack(results)


def main(
    model_name: str,
    model_repository_bucket: str,
    cluster_name: str,
    cluster_zone: str,
    num_nodes: int = 1,
    gpus_per_node: int = 4,
    vcpus_per_node: int = 16,
    gpu_type: str = "t4",
    batch_size: int = 4,
    num_samples: int = 5008,
    credentials: Optional[str] = None,
) -> None:
    repo = export(model_repository_bucket, model_name, batch_size)

    cluster = build_cluster(
        cluster_name,
        cluster_zone,
        num_nodes,
        gpus_per_node,
        vcpus_per_node,
        gpu_type,
        credentials,
    )

    try:
        deployment, load_balancer = cluster.deploy(
            "tritonserver.yaml",
            name="tritonserver",
            replicas=num_nodes,
            gpus=gpus_per_node,
            gpuType="t4",
            modelRepo=model_repository_bucket,
            tag="20.11",
        )

        deployment.wait_for_ready()
        server_url = f"{load_balancer.ip}:8001"

        dataset = np.random.randn(num_samples, 3, 224, 224).astype("float32")
        results = do_inference(
            dataset, server_url, model_name, batch_size=batch_size
        )
    finally:
        cluster.remove()
        repo.delete()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export a model and serve a model " "using an inference service"
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name to give the exported model",
    )
    parser.add_argument(
        "--model-repository-bucket",
        type=str,
        required=True,
        help="Name of the GCS bucket to store the model repo",
    )

    main("my-model", "hermes-hello-world", "hermes-hw-cluster", "us-east4-b")
