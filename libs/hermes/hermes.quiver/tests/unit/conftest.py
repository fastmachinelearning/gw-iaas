import random
from unittest.mock import Mock

import pytest

from hermes.quiver import ModelRepository
from hermes.quiver.io import GCSFileSystem, LocalFileSystem


@pytest.fixture(
    scope="module",
    params=[
        LocalFileSystem,
        pytest.param(GCSFileSystem, marks=pytest.mark.gcs),
    ],
)
def fs(request):
    return request.param


@pytest.fixture(scope="module")
def temp_fs(fs):
    filesystem = fs("hermes-quiver-test")
    yield filesystem
    filesystem.delete()


@pytest.fixture(scope="module")
def temp_repo(fs):
    repo = Mock()
    repo.fs = fs("hermes-quiver-test")
    yield repo
    repo.fs.delete()


@pytest.fixture(scope="module")
def temp_local_repo():
    repo = ModelRepository("hermes-quiver-test")
    yield repo
    repo.delete()


@pytest.mark.tensorflow
@pytest.fixture(scope="module")
def tf():
    import tensorflow as tf

    return tf


@pytest.fixture(scope="module", params=[10])
def torch_model(request):
    import torch

    class Model(torch.nn.Module):
        def __init__(self, size: int = 10):
            super().__init__()
            self.W = torch.eye(size)

        def forward(self, x):
            return torch.matmul(x, self.W)

    return Model(request.param)


@pytest.fixture(scope="module", params=[10])
def keras_model(request, tf):
    scope = "".join(random.choices("abcdefghijk", k=10))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                request.param,
                use_bias=False,
                kernel_initializer="identity",
                name=f"{scope}_dense",
            )
        ],
        name=scope,
    )

    # do a couple batch sizes to get variable size
    for batch_size in range(1, 3):
        y = model(tf.ones((batch_size, request.param)))
        assert (y.numpy() == 1.0).all()
    return model
