import pytest

from cake.targets import get_target
from cake.workloads import get_workload


@pytest.fixture(scope="session")
def km():
    return get_workload("kmeans_assign")


@pytest.fixture(scope="session")
def b200():
    return get_target("sm_100a")


SMALL = {"B": 2, "N": 512, "K": 512, "D": 128}
TAIL = {"B": 1, "N": 300, "K": 300, "D": 128}
