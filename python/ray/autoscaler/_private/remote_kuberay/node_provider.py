import datetime
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Set, Tuple

from kubernetes import client as k8s_client, config as k8s_config
from kubernetes.client import api_client as k8s_api_client, exceptions as k8s_exceptions
from kubernetes.client import configuration as k8s_client_config
from ray.tests.test_autoscaler import (
    DISABLE_LAUNCH_CONFIG_CHECK_KEY,
    DISABLE_NODE_UPDATERS_KEY,
    FOREGROUND_NODE_LAUNCH_KEY,
)
import requests

from ray.autoscaler._private.constants import WORKER_LIVENESS_CHECK_KEY
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
    BatchingNodeProvider,
    NodeData,
)
from ray.autoscaler.tags import (
    NODE_KIND_HEAD,
    NODE_KIND_WORKER,
    STATUS_UP_TO_DATE,
    STATUS_UPDATE_FAILED,
)

if TYPE_CHECKING:
    from kubernetes.client.models.v1_pod import V1Pod

KUBERAY_CRD_API_GROUP: Final[str] = "ray.io"
KUBERAY_CRD_VERSION: Final[str] = "v1"
KUBERAY_CLUSTER_CRD_PLURAL: Final[str] = "rayclusters"

# Key for KubeRay label that identifies a Ray pod as head or worker.
KUBERAY_LABEL_KEY_KIND: Final[str] = "ray.io/node-type"
# Key for KubeRay label that identifies the worker group (autoscaler node type) of a
# Ray pod.
KUBERAY_LABEL_KEY_TYPE: Final[str] = "ray.io/group"
# Key for KubeRay label that identifies the Ray cluster name of a Ray pod.
KUBERAY_LABEL_KEY_CLUSTER: Final[str] = "ray.io/cluster"

# These should be synced with:
# https://github.com/ray-project/kuberay/blob/f2d94ffe213dd8f69481b09c474047cb899fa73b/ray-operator/apis/ray/v1/raycluster_types.go#L165-L171 # noqa
# Kind label value indicating the pod is the head.
KUBERAY_KIND_HEAD: Final[str] = "head"
# Kind label value indicating the pod is the worker.
KUBERAY_KIND_WORKER: Final[str] = "worker"

KUBERAY_REQUEST_TIMEOUT_S: Final[Optional[int]] = int(
    os.getenv("KUBERAY_REQUEST_TIMEOUT_S", 60)
)

RAY_HEAD_POD_NAME: Final[Optional[str]] = os.getenv("RAY_HEAD_POD_NAME")

# https://kubernetes.io/docs/tasks/run-application/access-api-from-pod
# While running in a Pod, your container can create an HTTPS URL for the
# Kubernetes API server by fetching the KUBERNETES_SERVICE_HOST and
# KUBERNETES_SERVICE_PORT_HTTPS environment variables.
KUBERNETES_SERVICE_HOST: Final[str] = os.getenv(
    "KUBERNETES_SERVICE_HOST", "https://kubernetes.default"
)
KUBERNETES_SERVICE_PORT: Final[str] = os.getenv("KUBERNETES_SERVICE_PORT_HTTPS", "443")
KUBERNETES_HOST: Final[str] = f"{KUBERNETES_SERVICE_HOST}:{KUBERNETES_SERVICE_PORT}"
# Key for GKE label that identifies which multi-host replica a pod belongs to
REPLICA_INDEX_KEY: Final[str] = "replicaIndex"

TOKEN_REFRESH_PERIOD: Final[datetime.timedelta] = datetime.timedelta(minutes=30)

IGNORED_SETTINGS: Final[Set[str]] = {
    WORKER_LIVENESS_CHECK_KEY,
    DISABLE_NODE_UPDATERS_KEY,
    DISABLE_LAUNCH_CONFIG_CHECK_KEY,
    FOREGROUND_NODE_LAUNCH_KEY,
}

# Design:

# Each modification the autoscaler wants to make is posted to the API server goal state
# (e.g. if the autoscaler wants to scale up, it increases the number of
# replicas of the worker group it wants to scale, if it wants to scale down
# it decreases the number of replicas and adds the exact pods that should be
# terminated to the scaleStrategy).

# KubeRayNodeProvider inherits from BatchingNodeProvider.
# Thus, the autoscaler's create and terminate requests are batched into a single
# Scale Request object which is submitted at the end of autoscaler update.
# KubeRay node provider converts the ScaleRequest into a RayCluster CR patch
# and applies the patch in the submit_scale_request method.

# To reduce potential for race conditions, KubeRayNodeProvider
# aborts the autoscaler update if the operator has not yet processed workersToDelete -
# see KubeRayNodeProvider.safe_to_scale().
# Once it is confirmed that workersToDelete have been cleaned up, KubeRayNodeProvider
# clears the workersToDelete list.


# Note: Log handlers set up in autoscaling monitor entrypoint.
logger = logging.getLogger(__name__)


def node_data_from_pod(pod: "V1Pod") -> NodeData:
    """Converts a Ray Pod extracted from K8s into Ray NodeData.
    NodeData is processed by BatchingNodeProvider.
    """
    kind, type = kind_and_type(pod)
    status = status_tag(pod)
    ip = pod_ip(pod)
    replica_index = _replica_index_label(pod)
    return NodeData(
        kind=kind, type=type, replica_index=replica_index, status=status, ip=ip
    )


def kind_and_type(
    pod: "V1Pod",
) -> Tuple[NodeKind, NodeType]:
    """Determine Ray node kind (head or workers) and node type (worker group name)
    from a Ray pod's labels.
    """
    labels = pod.metadata.labels if pod.metadata and pod.metadata.labels else {}
    kind = (
        NODE_KIND_HEAD
        if labels.get(KUBERAY_LABEL_KEY_KIND) == KUBERAY_KIND_HEAD
        else NODE_KIND_WORKER
    )
    type = labels.get(KUBERAY_LABEL_KEY_TYPE)
    return kind, type


def _replica_index_label(pod: "V1Pod") -> Optional[str]:
    """Returns the replicaIndex label for a Pod in a multi-host TPU worker group.
    The replicaIndex label is set by the GKE TPU Ray webhook and is of
    the form {$WORKER_GROUP_NAME-$REPLICA_INDEX} where $REPLICA_INDEX
    is an integer from 0 to Replicas-1.
    """
    labels = pod.metadata.labels if pod.metadata and pod.metadata.labels else {}
    return labels.get(REPLICA_INDEX_KEY, None)


def pod_ip(pod: "V1Pod") -> NodeIP:
    return (
        pod.status.pod_ip if pod.status and pod.status.pod_ip else "IP not yet assigned"
    )


def status_tag(pod: "V1Pod") -> NodeStatus:
    """Convert Pod state to Ray autoscaler node status.

    See the doc string of the class
    batching_node_provider.NodeData for the semantics of node status.
    """
    if (
        not pod.status
        or not pod.status.container_statuses
        or not pod.status.container_statuses[0].state
    ):
        return "waiting"

    state = pod.status.container_statuses[0].state

    if state.running:
        return STATUS_UP_TO_DATE
    if state.waiting:
        return "waiting"
    if state.terminated:
        return STATUS_UPDATE_FAILED
    raise ValueError(f"Unexpected container state: {state}")


def worker_delete_patch(group_index: str, workers_to_delete: List[NodeID]):
    path = f"/spec/workerGroupSpecs/{group_index}/scaleStrategy"
    value = {"workersToDelete": workers_to_delete}
    return replace_patch(path, value)


def worker_replica_patch(group_index: str, target_replicas: int):
    path = f"/spec/workerGroupSpecs/{group_index}/replicas"
    value = target_replicas
    return replace_patch(path, value)


def replace_patch(path: str, value: Any) -> Dict[str, Any]:
    return {"op": "replace", "path": path, "value": value}


def load_k8s_config(
    config_file: Optional[pathlib.Path], context: Optional[str]
) -> k8s_client_config.Configuration:
    """
    Returns a k8s_client.Configuration object of the specified context
    or current context from the kubeconfig file.

    Args:
        config_file: The path to the kubeconfig file. If not provided,
            the default location (~/.kube/config) is used.
        context: The context to use from the kubeconfig file. If not provided,
            the current context is used.

    Returns:
        config: k8s_client.Configuration
    """

    # Create a new Configuration object
    config = k8s_client_config.Configuration()

    # Load the kube config from the default location (~/.kube/config) into the
    # Configuration object
    try:
        k8s_config.load_kube_config(
            config_file=config_file, context=context, client_configuration=config
        )
    except k8s_config.config_exception.ConfigException as e:
        raise ValueError(
            "Failed to load kube config. Try setting `.provider.context` in "
            "your cluster config YAML or checking the value is correct."
        ) from e

    return config


def load_k8s_info(
    config_file: Optional[pathlib.Path], context: Optional[str]
) -> Tuple[str, Dict[str, str], str]:
    """
    Loads info needed to access K8s resources.

    Returns:
        host: Host of the K8s API server
        headers: Headers with K8s access token
        verify: Path to certificate
    """

    # Create a new Configuration object
    config = k8s_client.Configuration()

    # Load the kube config from the default location (~/.kube/config) into the Configuration object
    try:
        k8s_config.load_kube_config(
            config_file=config_file, context=context, client_configuration=config
        )
    except k8s_config.config_exception.ConfigException as e:
        raise ValueError(
            "Failed to load kube config. Try setting `.provider.context` in your cluster config YAML or checking the value is correct."
        ) from e

    if not config.api_key.get("authorization"):
        raise ValueError("No authorization token found in kube config")

    if not config.ssl_ca_cert or not pathlib.Path(config.ssl_ca_cert).is_file():
        raise ValueError("No SSL CA certificate file found with kube config")

    headers = {
        "Authorization": str(config.api_key.get("authorization")),
    }

    return config.host, headers, config.ssl_ca_cert


def url_from_resource(
    namespace: str,
    path: str,
    kuberay_crd_version: str = KUBERAY_CRD_VERSION,
    kubernetes_host: str = KUBERNETES_HOST,
) -> str:
    """Convert resource path to REST URL for Kubernetes API server.

    Args:
        namespace: The K8s namespace of the resource
        path: The part of the resource path that starts with the resource type.
            Supported resource types are "pods" and "rayclusters".
        kuberay_crd_version: The API version of the KubeRay CRD.
            Looks like "v1alpha1", "v1".
        kubernetes_host: The host of the Kubernetes API server.
            Uses $KUBERNETES_SERVICE_HOST and
            $KUBERNETES_SERVICE_PORT to construct the kubernetes_host if not provided.

            When set by Kubernetes,
            $KUBERNETES_SERVICE_HOST could be an IP address. That's why the https
            scheme is added here.

            Defaults to "https://kubernetes.default:443".
    """
    if kubernetes_host.startswith("http://"):
        raise ValueError("Kubernetes host must be accessed over HTTPS.")
    if not kubernetes_host.startswith("https://"):
        kubernetes_host = "https://" + kubernetes_host
    if path.startswith("pods"):
        api_group = "/api/v1"
    elif path.startswith(KUBERAY_CLUSTER_CRD_PLURAL):
        api_group = f"/apis/{KUBERAY_CRD_API_GROUP}/{kuberay_crd_version}"
    else:
        raise NotImplementedError("Tried to access unknown entity at {}".format(path))
    return kubernetes_host + api_group + "/namespaces/" + namespace + "/" + path


def _worker_group_index(raycluster: Dict[str, Any], group_name: str) -> int:
    """Extract worker group index from RayCluster."""
    group_names = [
        spec["groupName"] for spec in raycluster["spec"].get("workerGroupSpecs", [])
    ]
    return group_names.index(group_name)


def _worker_group_max_replicas(
    raycluster: Dict[str, Any], group_index: int
) -> Optional[int]:
    """Extract the maxReplicas of a worker group.

    If maxReplicas is unset, return None, to be interpreted as "no constraint".
    At time of writing, it should be impossible for maxReplicas to be unset, but it's
    better to handle this anyway.
    """
    return raycluster["spec"]["workerGroupSpecs"][group_index].get("maxReplicas")


def _worker_group_replicas(raycluster: Dict[str, Any], group_index: int):
    # 1 is the default replicas value used by the KubeRay operator
    return raycluster["spec"]["workerGroupSpecs"][group_index].get("replicas", 1)


class RemoteKubeRayNodeProvider(BatchingNodeProvider):  # type: ignore
    def __init__(
        self,
        provider_config: Dict[str, Any],
        cluster_name: str,
    ):
        logger.info("Creating RemoteKubeRayNodeProvider")
        self.kube_config = (
            pathlib.Path(provider_config["kube_config"])
            # TODO (dxia): test setting this config
            if provider_config.get("kube_config", None)
            else None
        )
        self.context = provider_config.get("context", None)
        self.namespace = provider_config.get("namespace", "default")
        self.cluster_name = cluster_name

        self.k8s_api_client2 = k8s_api_client.ApiClient(
            load_k8s_config(config_file=self.kube_config, context=self.context)
        )

        if bool(set(provider_config.keys()) & IGNORED_SETTINGS):
            logger.warning(
                f"Ignoring {', '.join(IGNORED_SETTINGS)} in cluster config file. "
                "The remote_kuberay provider does not support these."
            )

        provider_config_copy = provider_config.copy()
        provider_config_copy[WORKER_LIVENESS_CHECK_KEY] = False
        provider_config_copy[DISABLE_NODE_UPDATERS_KEY] = True
        provider_config_copy[DISABLE_LAUNCH_CONFIG_CHECK_KEY] = True
        provider_config_copy[FOREGROUND_NODE_LAUNCH_KEY] = True
        BatchingNodeProvider.__init__(self, provider_config_copy, cluster_name)

    def get_node_data(self) -> Dict[NodeID, NodeData]:
        """Queries K8s for Pods in the RayCluster. Converts that Pod data into a
        map of Pod name to Ray NodeData, as required by BatchingNodeProvider.
        """
        try:
            # Store the RayCluster K8s custom resource
            self._raycluster = k8s_client.CustomObjectsApi(
                api_client=self.k8s_api_client2
            ).get_namespaced_custom_object(
                group=KUBERAY_CRD_API_GROUP,
                version=KUBERAY_CRD_VERSION,
                plural=KUBERAY_CLUSTER_CRD_PLURAL,
                namespace=self.namespace,
                name=self.cluster_name,
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"Ray cluster {self.cluster_name} not found in namespace {self.namespace} in context {self.context}."
                ) from e
            raise

        # If the Ray head Pod is specified, get its resource version.
        # Specifying a resource version in list requests is important for scalability:
        # https://kubernetes.io/docs/reference/using-api/api-concepts/#semantics-for-get-and-list
        resource_version = self._get_pods_resource_version()
        if resource_version:
            logger.info(
                f"Listing Pods for RayCluster {self.cluster_name}"
                f" in namespace {self.namespace}"
                f" at Pod's resource version >= {resource_version}."
            )

        # Filter Pods by cluster_name
        resource_version_match = "NotOlderThan" if resource_version else None
        pod_list = k8s_client.CoreV1Api(
            api_client=self.k8s_api_client2
        ).list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"{KUBERAY_LABEL_KEY_CLUSTER}={self.cluster_name}",
            resource_version=resource_version,
            resource_version_match=resource_version_match,
        )

        assert pod_list.metadata
        logger.info(
            f"Fetched Pod data at resource version {pod_list.metadata.resource_version}."
        )

        # Extract node data from the Pod list.
        node_data_dict = {}
        for pod in pod_list.items:
            assert pod.metadata
            # Kubernetes sets metadata.deletionTimestamp immediately after admitting a
            # request to delete an object. Full removal of the object may take some time
            # after the deletion timestamp is set. See link for details:
            # https://kubernetes.io/docs/reference/using-api/api-concepts/#resource-deletion
            if pod.metadata.deletion_timestamp:
                # Ignore Pods marked for termination.
                continue
            pod_name = pod.metadata.name
            assert pod_name
            node_data_dict[pod_name] = node_data_from_pod(pod)

        # TODO (dxia): This function is called twice when you run `ray get-head-ip desired.yaml`.
        # It's called once for each worker if you run `ray get-worker-ips desired.yaml`.
        # This is because python/ray/autoscaler/_private/commands.py get_head_node_ip()
        # ends up calling this function twice. Ideally this would only be called once because
        # the return value has all the IPs of all Pods.
        # print(f"NODE_DATA_DICT: {node_data_dict}")
        return node_data_dict

    def external_ip(self, node_id: str) -> str:
        """Returns the external IP of the given node."""
        node = self.get_node_data().get(node_id)
        if not node or not node.ip:
            raise RuntimeError(f"Node {node_id} not found or IP not assigned.")
        return node.ip

    def _get_pods_resource_version(self) -> Optional[str]:
        """
        Get the Ray head Pod's resource version.
        Returns None if the head Pod is not specified.
        """
        if not RAY_HEAD_POD_NAME:
            return None

        try:
            pod = k8s_client.CoreV1Api(
                api_client=self.k8s_api_client2
            ).read_namespaced_pod(
                name=RAY_HEAD_POD_NAME,
                namespace=self.namespace,
            )
        except k8s_exceptions.ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Ray head Pod {RAY_HEAD_POD_NAME} not found. Ignoring specified Ray head Pod."
                )
                return None
            raise

        return pod.metadata.resource_version if pod.metadata else None
