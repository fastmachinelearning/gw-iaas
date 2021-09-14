import os

import pytest

from gravswell.cloudbreak import google as cb


def test_cluster_manager(zone="us-central1-f"):
    manager = cb.ClusterManager(zone=zone)
    print(manager.name)

    creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
    with pytest.raises(ValueError):
        manager = cb.ClusterManager(zone=zone)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
