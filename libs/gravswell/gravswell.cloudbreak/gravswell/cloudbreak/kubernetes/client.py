import re
import time
import typing
from base64 import b64decode
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import kubernetes
import requests
import yaml
from urllib3.exceptions import MaxRetryError

from gravswell.cloudbreak.utils import wait_for

if typing.TYPE_CHECKING:
    from gravswell.cloudbreak.base.kubernetes import Cluster


def _get_service_account_access_token():
    METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
    METADATA_HEADERS = {"Metadata-Flavor": "Google"}
    SERVICE_ACCOUNT = "default"

    url = "{}/instance/service-accounts/{}/token".format(
        METADATA_URL, SERVICE_ACCOUNT
    )

    # Request an access token from the metadata server.
    r = requests.get(url, headers=METADATA_HEADERS)
    r.raise_for_status()

    # Extract the access token from the response.
    return r.json()["access_token"]


class K8sApiClient:
    def __init__(self, cluster: "Cluster"):
        # TODO: generalize the initialization putting methods
        # on the `cluster` object to return the appropriate
        # information
        try:
            response = cluster.get()
        except requests.HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"Cluster {cluster} not currently deployed")
            raise

        # create configuration using bare minimum info
        configuration = kubernetes.client.Configuration()
        configuration.host = f"https://{response.endpoint}"

        with NamedTemporaryFile(delete=False) as ca_cert:
            certificate = response.master_auth.cluster_ca_certificate
            ca_cert.write(b64decode(certificate))
        configuration.ssl_ca_cert = ca_cert.name
        configuration.api_key_prefix["authorization"] = "Bearer"

        # get credentials for conencting to server
        # GCP code lifted from
        # https://cloud.google.com/compute/docs/access/
        # create-enable-service-accounts-for-instances#applications
        if cluster.client.credentials is None:
            access_token = _get_service_account_access_token()
            self._refresh = True
        else:
            access_token = cluster.client.credentials.token
            self._refresh = False

        configuration.api_key["authorization"] = access_token

        # return client instantiated with configuration
        self._client = kubernetes.client.ApiClient(configuration)

    @contextmanager
    def _maybe_refresh(self):
        body = {}
        try:
            yield body
        except kubernetes.client.exceptions.ApiException as e:
            body.update(yaml.safe_load(e.body))
            if e.status == 401:
                if self._refresh:
                    token = _get_service_account_access_token()
                    self.client.configuration.api_key["authorization"] = token
                else:
                    raise RuntimeError("Unauthorized request to cluster")

    def create_from_yaml(
        self,
        file: str,
        repo: typing.Optional[str] = None,
        branch: typing.Optional[str] = None,
        namespace: str = "default",
        ignore_if_exists: bool = True,
        **kwargs,
    ):
        with self._maybe_refresh() as body:
            # get deploy file content either from
            # local file or from github repo. Parse
            # yaml go template wildcards with kwargs
            content = get_content(file, repo, branch, **kwargs)

            failures = []
            k8s_objects = []
            for yml_document in yaml.safe_load_all(content):
                start_time = time.time()
                while (time.time() - start_time) < 10:
                    # sometimes trying to connect to a freshly
                    # created cluster can be a bit fickle,
                    # so use this timeout to catch the MaxRetryError
                    # a few times before raising an error
                    try:
                        # try to create k8s objects one at a time
                        created = kubernetes.utils.create_from_dict(
                            self._client, yml_document, namespace=namespace
                        )
                        k8s_objects.append(created)
                        break
                    except kubernetes.utils.FailToCreateError as failure:
                        # kubernetes exception in creation. Keep track
                        # of all that get raised and raise them at the end
                        for exc in failure.api_exceptions:
                            reason = yaml.safe_load(exc.body)["reason"]
                            if (
                                reason != "AlreadyExists"
                                or not ignore_if_exists
                            ):
                                # if the problem was that the object already
                                # exists and we indicated to ignore this
                                # problem, we don't need to append the
                                # exception
                                failures.append(exc)

                        # break here because there's no
                        # point in trying to create again
                        break
                    except MaxRetryError:
                        # catch this error a few times
                        # before we decide it's fatal
                        time.sleep(1)
                else:
                    # the object either couldn't create or hit
                    # an api exception before the timeout,
                    # so raise an error
                    raise RuntimeError("Encountered too many retries")

            if failures:
                raise kubernetes.utils.FailToCreateError(failures)

        if body:
            raise RuntimeError(f"Encountered exception {body}")
        return k8s_objects

    def remove_deployment(self, name: str, namespace: str = "default"):
        app_client = kubernetes.client.AppsV1Api(self._client)

        def _try_cmd(cmd):
            for _ in range(2):
                with self._maybe_refresh() as body:
                    cmd(name=name, namespace=namespace)
                if body and body["code"] == 404:
                    return True
                elif not body or body["code"] != 401:
                    break
            return False

        _try_cmd(app_client.delete_namespaced_deployment)

        def _deleted_callback():
            return _try_cmd(app_client.read_namespaced_deployment)

        wait_for(_deleted_callback, f"Waiting for deployment {name} to delete")

    def wait_for_deployment(self, name: str, namespace: str = "default"):
        app_client = kubernetes.client.AppsV1Api(self._client)

        _start_time = time.time()
        _grace_period_seconds = 10

        def _ready_callback():
            try:
                response = app_client.read_namespaced_deployment_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException as e:
                if e.status == 404:
                    raise RuntimeError(f"Deployment {name} no longer exists!")
                raise
            except MaxRetryError:
                time.sleep(1)
                return False

            conditions = response.status.conditions
            if conditions is None:
                return False
            statuses = {i.type: eval(i.status) for i in conditions}

            try:
                if statuses["Available"]:
                    return True
            except KeyError:
                if (time.time() - _start_time) > _grace_period_seconds:
                    raise ValueError("Couldn't find readiness status")

            try:
                if not statuses["Progressing"]:
                    raise RuntimeError(
                        f"Deployment {name} stopped progressing"
                    )
            except KeyError:
                pass
            finally:
                return False

        wait_for(_ready_callback, f"Waiting for deployment {name} to deploy")

    def wait_for_service(self, name: str, namespace: str = "default"):
        core_client = kubernetes.client.CoreV1Api(self._client)

        def _ready_callback():
            try:
                response = core_client.read_namespaced_service_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException as e:
                if e.status == 404:
                    raise RuntimeError(f"Service {name} no longer exists!")
                raise

            try:
                ip = response.status.load_balancer.ingress[0].ip
            except TypeError:
                return False
            return ip or False

        return wait_for(
            _ready_callback, f"Waiting for service {name} to be ready"
        )

    def wait_for_daemon_set(self, name: str, namespace: str = "kube-system"):
        core_client = kubernetes.client.CoreV1Api(self._client)

        def _ready_callback():
            try:
                response = core_client.read_namespaced_daemon_set_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException as e:
                if e.status == 404:
                    raise RuntimeError(f"Daemon set {name} no longer exists!")
                raise

            status = response.status
            return status.desired_number_scheduled == status.number_ready


def get_content(
    file: str,
    repo: typing.Optional[str] = None,
    branch: typing.Optional[str] = None,
    **kwargs,
):
    if repo is not None:
        if branch is None:
            # if we didn't specify a branch, default to
            # trying main first but try master next in
            # case the repo hasn't changed yet
            branches = ["main", "master"]
        else:
            # otherwise just use the specified branch
            branches = [branch]

        for branch in branches:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                content = response.content.decode()
                break
            except Exception:
                pass
        else:
            raise ValueError(
                "Couldn't find file {} at github repo {} "
                "in branches {}".format(file, repo, ", ".join(branches))
            )
    else:
        with open(file, "r") as f:
            content = f.read()

    return sub_values(content, **kwargs)


def sub_values(content: str, **kwargs):
    match_re = re.compile("(?<={{ .Values.)[a-zA-Z0-9]+?(?= }})")
    found = set()

    def replace_fn(match):
        varname = match_re.search(match.group(0)).group(0)
        found.add(varname)
        try:
            return str(kwargs[varname])
        except KeyError:
            raise ValueError(f"No value provided for wildcard {varname}")

    content = re.sub("{{ .Values.[a-zA-Z0-9]+? }}", replace_fn, content)

    missing = set(kwargs) - found
    if missing:
        raise ValueError(
            "Provided unused wildcard values: {}".format(", ".join(missing))
        )
    return content
