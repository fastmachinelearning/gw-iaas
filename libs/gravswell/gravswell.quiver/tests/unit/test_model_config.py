from dataclasses import dataclass

import pytest
from tritonclient.grpc.model_config_pb2 import ModelInstanceGroup

from gravswell.quiver import ModelConfig
from gravswell.quiver.io import FileSystem, GCSFileSystem, LocalFileSystem


class Convention:
    name = "onnxruntime_onnx"
    filename = "model.onnx"


class Platform:
    convention = Convention


@dataclass
class DummyModel:
    fs: FileSystem

    name = "test-model"
    platform = Platform


@pytest.mark.parametrize("fs_type", [LocalFileSystem, GCSFileSystem])
def test_model_config(fs_type):
    try:
        fs = fs_type("gravswell-quiver-test")
    except ValueError:
        fs = fs_type("gs://gravswell-quiver-test")
    fs.soft_makedirs(DummyModel.name)

    try:
        model = DummyModel(fs)
        config = ModelConfig(model, max_batch_size=8)
        assert config.name == "test-model"
        assert config.platform == "onnxruntime_onnx"
        assert config.max_batch_size == 8

        input = config.add_input(
            name="test_input", shape=(None, 8), dtype="float32"
        )
        assert len(config.input) == 1
        assert config.input[0] == input
        assert config.input[0].name == "test_input"

        instance_group = config.add_instance_group(kind="gpu", gpus=2, count=4)
        assert len(config.instance_group) == 1
        assert instance_group == config.instance_group[0]
        assert config.instance_group[0].kind == ModelInstanceGroup.KIND_GPU
        assert config.instance_group[0].gpus == [0, 1]
        assert config.instance_group[0].count == 4

        config.instance_group[0].count = 6
        assert config.instance_group[0].count == 6

        config.write()

        # first make sure that the the config
        # gotten written properly to the right
        # place
        config_path = model.fs.join(config.name, "config.pbtxt")
        new_config = model.fs.read_config(config_path)
        assert new_config.SerializeToString() == config.SerializeToString()

        # next initialize a new config using
        # the existing model to see if it loads
        # things in properly
        new_config = ModelConfig(model)
        assert new_config.SerializeToString() == config.SerializeToString()

        # finally check to make sure that kwargs
        # provided with initialization override
        # those in the existing config
        new_config = ModelConfig(model, max_batch_size=4)
        assert new_config.max_batch_size == 4
        assert new_config.input[0].name == "test_input"
    except Exception:
        raise
    finally:
        fs.delete()
