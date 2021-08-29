import logging
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import google
from google.auth.transport.requests import Request as AuthRequest
from google.cloud import container_v1 as container
from google.oauth2 import service_account

from gravswell.cloudbreak.kubernetes import K8sApiClient
from gravswell.cloudbreak.utils import wait_for

_credentials_type = Union[str, service_account.Credentials, None]


def snakeify(name: str) -> str:
    return re.sub("(?<!^)(?=[A-Z])", "_", name).lower()


def make_credentials(
    service_account_key_file: str,
    scopes: Optional[Sequence[str]] = None,
):
    """
    Cheap wrapper around service account creation
    class method to simplify a couple gotchas. Might
    either be overkill or may be better built as a
    class with more functionality, not sure yet.
    """
    scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_file(
        service_account_key_file,
        scopes=scopes,
    )
    credentials.refresh(AuthRequest())
    return credentials


class ThrottledClient:
    def __init__(
        self,
        credentials: Union[str, service_account.Credentials, None] = None,
        throttle_secs: float = 1.0,
    ):
        if isinstance(credentials, str):
            credentials = make_credentials(credentials)

        self.credentials = credentials
        self._client = container.ClusterManagerClient(
            credentials=self.credentials
        )
        self.throttle_secs = throttle_secs
        self._last_request_time = time.time()

    @property
    def client(self):
        return self

    @property
    def name(self):
        return ""

    def make_request(self, request, **kwargs):
        request_type = type(request).__name__.replace("Request", "")
        request_fn_name = snakeify(request_type)
        request_fn = getattr(self._client, request_fn_name)
        while (time.time() - self._last_request_time) < self.throttle_secs:
            time.sleep(0.01)
        return request_fn(request=request, **kwargs)


@dataclass
class Resource:
    name: str
    parent: "Resource"

    @property
    def client(self):
        return self.parent.client

    @property
    def resource_type(self):
        return type(self).__name__

    @classmethod
    def create(cls, resource, parent, **kwargs):
        resource_type = type(resource).__name__
        if resource_type == "Cluster":
            cls = Cluster
        elif resource_type == "NodePool":
            cls = NodePool
        else:
            raise TypeError(f"Unknown GKE resource type {resource_type}")

        obj = cls(resource.name, parent, **kwargs)
        create_request_cls = getattr(
            container, f"Create{obj.resource_type}Request"
        )

        resource_type = snakeify(obj.resource_type)
        kwargs = {resource_type: resource, "parent": parent.name}
        create_request = create_request_cls(**kwargs)
        try:
            obj.client.make_request(create_request)
        except google.api_core.exceptions.AlreadyExists:
            pass
        return obj

    def delete(self):
        delete_request_cls = getattr(
            container, f"Delete{self.resource_type}Request"
        )
        delete_request = delete_request_cls(name=str(self))
        return self.client.make_request(delete_request)

    def get(self, timeout=None):
        get_request_cls = getattr(container, f"Get{self.resource_type}Request")
        get_request = get_request_cls(name=str(self))

        try:
            return self.client.make_request(get_request, timeout=timeout)
        except google.api_core.exceptions.NotFound:
            raise ValueError(f"Couldn't get resource {self}")

    def _raise_bad_status(self, response):
        raise RuntimeError(
            f"Resource {self} reached status {response.status} "
            f"while deleting with conditions {response.conditions}"
        )

    def is_ready(self) -> bool:
        response = self.get(timeout=5)
        if response.status == 2:
            return True
        elif response.status > 2:
            self._raise_bad_status(response)
        return False

    def submit_delete(self) -> bool:
        """
        Attempt to submit a delete request for a resource.
        Returns `True` if the request is successfully
        submitted or if the resource can't be found,
        and `False` if the request can't be submitted
        """
        try:
            self.delete()
            return True
        except google.api_core.exceptions.NotFound:
            # resource is gone, so we're good
            return True
        except google.api_core.exceptions.BadRequest:
            # Resource is tied up, so indicate that
            # the user will need to try again later
            return False

    def is_deleted(self) -> bool:
        """
        check if a submitted delete request has completed
        """
        try:
            response = self.get(timeout=5)
        except ValueError as e:
            if str(e) != f"Couldn't get resource {self}":
                raise
            # couldn't find the resource, so assume
            # the deletion went off swimmingly
            return True
        if response.status > 5:
            self._raise_bad_status(response)
        return False

    def __str__(self):
        resource_type = self.resource_type
        camel = resource_type[0].lower() + resource_type[1:]
        return self.parent.name + "/{}/{}".format(camel, self.name)

    @property
    def message(self):
        resource_type = snakeify(type(self).__name__).replace("_", " ")
        return " ".join([resource_type, self.name])


@dataclass
class NodePool(Resource):
    """
    Args:
        wait:
            How long to wait for the request resources
            to become available if a stockout is encountered.
            Default `None` indicates raise an error as soon
            as a stockout is encountered.
    """

    wait: Optional[float] = None

    def __post_init__(self):
        self._init_time = time.time()

    def is_ready(self):
        response = self.get(timeout=5)
        if response.status == 2:
            # the resource is ready
            return True
        elif response.status == 6:
            # something has gone wrong, see what the
            # primary condition code is
            code = response.conditions[0].code

            if code != container.StatusCondition.Code.GCE_STOCKOUT:
                # if it's not a stockout issue, raise
                # the error
                self._raise_bad_status(response)
            elif (
                self.wait is None
                or (time.time() > self._init_time) < self.wait
            ):
                # if we indicated not to wait or have exhausted the
                # amount of time we were willing to wait, raise an error
                raise RuntimeError(
                    f"Resource {self} encountered GCE stockout "
                    "on creation and timed out"
                )
        elif response.status > 2:
            # for any other response other than
            # success, raise an error
            self._raise_bad_status(response)
        else:
            # the status is < 2, which means the node
            # pool is still being generated, so be patient
            return False


@dataclass
class ManagerResource(Resource):
    def __post_init__(self):
        self._resources = {}

        mrts = self.managed_resource_type.__name__ + "s"
        snaked = snakeify(mrts)

        list_request_cls = getattr(container, f"List{mrts}Request")
        list_resource_request = list_request_cls(parent=str(self))
        list_resource_fn = getattr(self.client._client, f"list_{snaked}")

        try:
            response = list_resource_fn(list_resource_request)
        except google.api_core.exceptions.NotFound:
            return

        resources = getattr(response, snaked)
        for resource in resources:
            self._resources[resource.name] = self.managed_resource_type(
                resource.name, self
            )

    @property
    def managed_resource_type(self):
        raise NotImplementedError

    @property
    def resources(self):
        return self._resources.copy()

    def create_resource(self, resource):
        if type(resource).__name__ != self.managed_resource_type.__name__:
            raise TypeError(
                "{} cannot manage resource {}".format(
                    type(self).__name__, type(resource).__name__
                )
            )

        resource = Resource.create(resource, self)

        wait_for(
            resource.is_ready,
            f"Waiting for {resource.message} to become ready",
        )
        self._resources[resource.name] = resource
        return resource

    def delete_resource(self, resource):

        wait_for(
            resource.submit_delete,
            f"Waiting for {resource.message} to become available to delete",
        )

        wait_for(
            resource.is_deleted, f"Waiting for {resource.message} to delete"
        )
        self._resources.pop(resource.name)

    @contextmanager
    def manage(self, resource, keep=False, **kwargs):
        resource = self.create_resource(resource, **kwargs)

        try:
            yield resource
        except Exception as e:
            if not keep:
                logging.error(
                    f"Encountered error, removing {resource.message}: {e}"
                )
            raise
        finally:
            if not keep:
                self.delete_resource(resource)


@dataclass
class Cluster(ManagerResource):
    def __post_init__(self):
        self._k8s_client = None
        super().__post_init__()

    @property
    def managed_resource_type(self):
        return NodePool

    @property
    def k8s_client(self):
        # try to create the client this way because otherwise we
        # would need to wait until the cluster is ready at
        # initialization time in order to get the endpoint. If you're
        # not going to call `wait_for(cluster.is_ready)`, make sure to
        # wrap this in a catch for a RuntimeError
        # TODO: is it worth starting to introduce custom errors here
        # to make catching more intelligible?
        if self._k8s_client is None:
            self._k8s_client = K8sApiClient(self)
        return self._k8s_client

    def deploy(
        self,
        file: str,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
        namespace: str = "default",
        ignore_if_exists: bool = True,
        **kwargs,
    ):
        return self.k8s_client.create_from_yaml(
            file, repo, branch, namespace, ignore_if_exists, **kwargs
        )

    def remove_deployment(self, name: str, namespace: str = "default"):
        return self.k8s_client.remove_deployment(name, namespace)

    def deploy_gpu_drivers(self) -> None:
        self.deploy(
            "nvidia-driver-installer/cos/daemonset-preloaded.yaml",
            repo="GoogleCloudPlatform/container-engine-accelerators",
            branch="master",
            ignore_if_exists=True,
        )
        self.k8s_client.wait_for_daemon_set(name="nvidia-driver-installer")


class ClusterManager(ManagerResource):
    def __init__(
        self,
        project: str,
        zone: str,
        credentials: Union[str, service_account.Credentials, None] = None,
    ) -> None:
        parent = ThrottledClient(credentials)
        name = f"projects/{project}/locations/{zone}"
        super().__init__(name, parent)

    @property
    def managed_resource_type(self):
        return Cluster

    def __str__(self):
        return self.name


def create_gpu_node_pool_config(
    vcpus: int, gpus: int, gpu_type: str, **kwargs
) -> container.NodeConfig:
    if (math.log2(vcpus) % 1 != 0 and vcpus != 96) or vcpus > 96:
        raise ValueError(f"Can't configure node pool with {vcpus} vcpus")

    if gpus < 1 or gpus > 8:
        raise ValueError(f"Can't configure node pool with {gpus} gpus")

    if gpu_type not in ["t4", "v100", "p100", "p4", "k80"]:
        raise ValueError(
            "Can't configure n1 standard node pool "
            f"with unknown gpu type {gpu_type}"
        )

    return container.NodeConfig(
        machine_type=f"n1-standard-{vcpus}",
        oauth_scopes=[
            "https://www.googleapis.com/auth/devstorage.read_only",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring",
            "https://www.googleapis.com/auth/service.management.readonly",
            "https://www.googleapis.com/auth/servicecontrol",
            "https://www.googleapis.com/auth/trace.append",
        ],
        accelerators=[
            container.AcceleratorConfig(
                accelerator_count=gpus,
                accelerator_type=f"nvidia-tesla-{gpu_type}",
            )
        ],
        **kwargs,
    )
