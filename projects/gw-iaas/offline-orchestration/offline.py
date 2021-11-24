import logging
import math
import os
import sys
from typing import List, Optional, Tuple, Union

from herems.quiver.io import GCSFileSystem

from hermes.cloudbreak.clouds import google as cb
from hermes.quiver.model_repository import ModelRepository
from hermes.stillwater import ServerMonitor
from hermes.typeo import typeo

base_cmd = "source $CONDA_INIT && conda activate gwftools && \\"
deepclean_cmd = (
    base_cmd
    + r"""
poetry run deepclean \
    --data-dir gs://{input_data_bucket_name} \
    --write-dir gs://{output_data_bucket_name} \
    --kernel-length 1 \
    --stride-length {kernel_stride} \
    --sample-rate 4096 \
    --inference-rate {throughput_per_client} \
    --channels channels.deepclean.txt \
    --sequence_id {{sequence_id}} \
    --url {{ip}}:8001 \
    --model-name {model_name} \
    --model-version 1 \
    --t0 {{t0}} \
    --length {{length}} \
    --preprocess_pkl ppr.pkl \
    --log-file log.txt
"""
)


def divvy_up_files(
    bucket_name: str, total_clients: int, split_files: bool
) -> Tuple[List[float], List[float]]:
    bucket_name = bucket_name.replace("gs://", "")
    fs = GCSFileSystem(bucket_name)

    fnames = [f for f in fs.list() if f.endswith(".gwf")]
    if len(fnames) == 0:
        dirs = fnames
        fnames = []
        for d in dirs:
            fnames.extend([f for f in fs.list(d) if f.endswith(".gwf")])

    if len(fnames) == 0:
        try:
            bucket_name, root = bucket_name.split("/", maxsplit=1)
        except TypeError:
            root = ""

        raise ValueError(
            "No .gwf files found at first or second "
            "level of directory '{}' in bucket '{}'".format(bucket_name, root)
        )

    t0s, lengths = [], []
    for f in fnames:
        f = f.replace(".gwf", "")
        t0, length = tuple(map(int, f.split("-")[-2:]))
        t0s.append(t0)
        lengths.append(length)
    total_length = sum(lengths)

    # if we're willing to have different clients process
    # chunks of the same file, then divvy up the total
    # time evenly among all the clients and let them
    # sort it out. Add 1 to account for the fact that the
    # first second each client processes will have an
    # incorrect initial state
    if split_files:
        return t0s, [total_length / total_clients + 1 for _ in t0s]

    # otherwise break up the files as evenly
    # as possible, assigning whatever remainder
    # is left to the first `leftover` clients (which
    # will therefore be processing one file more than
    # the other clients)
    div, leftover = divmod(len(fnames), total_clients)
    groups, idx = [], 0
    while idx < len(fnames):
        length = div
        if leftover > 0:
            length += 1
            leftover -= 1

        slc = slice(idx, idx + length)
        groups.append((t0s[slc], lengths(slc)))
        idx += length

    # we appended things in the transposed
    # order that we want, so transpose here
    return list(zip(*groups))


def scale_models(
    model_repo_bucket_name: str,
    model_name: str,
    instances_per_gpu: Union[int, dict],
    num_gpus: int,
    num_clients: int,
) -> None:
    # load in the model repository and make sure that
    # the desired model exists and scale the number
    # of instances of all its constituent models
    repo = ModelRepository("gs://" + model_repo_bucket_name)

    def _get_model(model_name):
        try:
            return repo.models[model_name]
        except KeyError:
            raise ValueError(
                "No model named '{}' in model repo bucket '{}'".format(
                    model_name, model_repo_bucket_name
                )
            )

    model = _get_model(model_name)
    try:
        # check if this is an ensemble model
        steps = model.config.ensemble_scheduling.step
    except AttributeError:
        # this isn't an ensemble model, so just scale
        # the indicated model
        models = [model_name]
    else:
        # scale each of the constituent models in the ensemble
        models = [i.model_name for i in steps]

    if not isinstance(instances_per_gpu, dict):
        # if we didn't specify a number of instances for
        # each model, assume that we specified a single
        # number for _all_ models
        instances_per_gpu = {i: instances_per_gpu for i in models}

    # set the scale for the snapshotter model separately
    # if it exists by making sure that we'll have enough
    # available snapshot instances to accommodate all
    # of our client streams
    try:
        snapshotter = [i for i in models if i.startswith("snapshotter")][0]
    except IndexError:
        pass
    else:
        instances_per_gpu[snapshotter] = math.ceil(num_clients / num_gpus)

    # for each model, adjust its instance group to
    # match the indicated number of instances per GPU
    for model_name, instances in instances_per_gpu:
        model = _get_model(model_name)

        # try to scale an existing instance group,
        # and add one if it fails
        try:
            model.config.scale_instance_group(instances)
        except ValueError:
            model.config.add_instance_group(count=instances)

        # write the updated config to the model repository
        model.config.write()


@typeo("Offline processing")
def main(
    run_dir: str,
    username: str,
    ssh_key_file: str,
    project: str,
    zone: str,
    cluster_name: str,
    vm_name: str,
    input_data_bucket_name: str,
    output_data_bucket_name: str,
    model_repo_bucket_name: str,
    num_server_nodes: int,
    gpus_per_server_node: int,
    clients_per_server_node: int,
    instances_per_gpu: Union[int, dict],
    vcpus_per_server_gpu: int,
    model_name: str,
    kernel_stride: float,
    vcpus_per_client: int = 4,
    credentials: Optional[str] = None,
    gpu_type: str = "t4",
    split_files: str = False,
    throughput_per_client: float = 1000,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    # set up host-machine logging
    if log_file is None:
        kwargs = {"stream": sys.stdout}
    else:
        kwargs = {"file": log_file}

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        **kwargs,
    )

    # inspect the input data bucket to assign processing
    # start times and lengths to each client
    total_clients = clients_per_server_node * num_server_nodes
    t0s, lengths = divvy_up_files(
        input_data_bucket_name, split_files, total_clients
    )

    # scale up each individual model to the indicated level
    scale_models(
        model_repo_bucket_name,
        model_name,
        instances_per_gpu,
        num_gpus=gpus_per_server_node * num_server_nodes,
        num_clients=total_clients,
    )

    # build the resources to create a cluster and
    # gpu-enabled nodes on it
    cluster_manager = cb.ClusterManager(
        project=project, zone=zone, credentials=credentials
    )
    cluster_config = cb.kubernetes.container.Cluster(
        name=cluster_name,
        node_pools=[
            cb.kubernetes.container.NodePool(
                name="default-pool",
                initial_node_count=1,
                config=cb.kubernetes.container.NodeConfig(),
            )
        ],
    )
    cluster_node_pool_config = cb.kubernetes.container.NodePool(
        name="tritonserver-{}-pool".format(gpu_type),
        initial_node_count=num_server_nodes,
        config=cb.create_gpu_node_pool_config(
            vcpus=vcpus_per_server_gpu * gpus_per_server_node,
            gpus=gpus_per_server_node,
            gpu_type=gpu_type,
        ),
    )

    # now build the resources for the client VMs
    client_config = cb.make_simple_debian_instance_description(
        name=vm_name,
        zone=zone,
        vcpus=vcpus_per_client,
        service_account=(
            credentials or os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        ),
    )
    client_manager = cb.VMManager(
        description=client_config, credentials=credentials
    )

    # create the VMs in the background then
    # create the cluster and node pools before
    # checking to make sure all the VMs are ready
    logging.info(
        "Configuring {} client VMs with {} vCPUs each".format(
            total_clients, vcpus_per_client
        )
    )
    client_manager.create(total_clients, username, ssh_key_file)
    try:
        with cluster_manager.add(cluster_config) as cluster:
            # GCP install GPU drivers as a *daemonset* on the
            # cluster, meaning this container which exposes the
            # drivers to host containers will be automatically
            # deployed on any nodes we create afterwards
            cluster.deploy_gpu_drivers()

            # deploy a pool of GPU-enabled nodes in this cluster
            node_config = cluster_node_pool_config.config
            logging.info(
                "Configuring {} server nodes with {} vCPUs"
                " and {} {} GPUs each".format(
                    cluster_node_pool_config.initial_node_count,
                    node_config.machine_type.split("-")[-1],
                    node_config.accelerators[0].accelerator_count,
                    node_config.accelerators[0].accelerator_type.split("-")[
                        -1
                    ],
                )
            )
            with cluster.add(cluster_node_pool_config):
                # note that we'll need a separate deployment
                # and load balancer for _each cluster node_,
                # since we need streaming state updates to
                # get routed to the same snapshotter instance
                deployments, services = [], []
                for i in range(num_server_nodes):
                    deployment, service = cluster.deploy(
                        "tritonserver.yaml",
                        num_gpus=gpus_per_server_node,
                        tag="20.11",
                        vcpus=vcpus_per_server_gpu * gpus_per_server_node,
                        bucket=model_repo_bucket_name,
                        name=f"tritonserver-{i}",
                    )
                    deployments.append(deployment)
                    services.append(service)

                # now that we have everything we need
                # deployed, wait for it all to indicate
                # that it's ready to go before proceeding
                client_manager.wait_for_ready()
                for deployment, service in zip(deployments, service):
                    deployment.wait_for_ready()
                    service.wait_for_ready()

                # assign each client node its own load
                # balancer IP to make requests to
                ips = [service.ip for service in services]
                ip_repeats = math.ceil(total_clients / len(ips))
                ips = ([ips] * ip_repeats)[:total_clients]

                # monitor the throughput on all the server
                # nodes in a separate process while the
                # clients run and save the results to a csv
                results_file = os.path.join(run_dir, "results.csv")
                logging.info(
                    f"Starting clients and writing results to {results_file}"
                )
                with ServerMonitor(model_name, ips, results_file):
                    stdouts, stderrs = client_manager.run(
                        deepclean_cmd.format(
                            input_data_bucket_name=input_data_bucket_name,
                            output_data_bucket_name=output_data_bucket_name,
                            kernel_stride=kernel_stride,
                            throughput_per_client=throughput_per_client,
                            model_name=model_name,
                        ),
                        sequence_id=[1001 + i for i in range(total_clients)],
                        ip=ips,
                        t0=t0s,
                        length=lengths,
                    )

                # copy all the logs from client nodes to our
                # run directory for reference
                log_dir = os.path.join(run_dir, "logs")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                for name, vm in client_manager.resources:
                    vm.scp("log.txt", os.path.join(log_dir, f"{name}.log"))
    finally:
        # delete our VMs no matter what happened
        client_manager.delete()
        client_manager.wait_for_delete()


if __name__ == "__main__":
    main()
