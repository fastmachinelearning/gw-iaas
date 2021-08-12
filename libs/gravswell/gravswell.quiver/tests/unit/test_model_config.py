from tritonclient.grpc.model_config_pb2 import ModelInstanceGroup

from gravswell.quiver import ModelConfig
from gravswell.quiver.io import LocalFileSystem


class Convention:
    name = "onnxruntime_onnx"
    filename = "model.onnx"


class Platform:
    convention = Convention


class DummyModel:
    name = "test-model"
    fs = LocalFileSystem("gravswell-quiver-test")
    platform = Platform


def test_model_config():
    DummyModel.fs.soft_makedirs(DummyModel.name)

    try:
        config = ModelConfig(DummyModel, max_batch_size=8)
        assert config.name == "test-model"
        assert config.platform == "onnxruntime_onnx"
        assert config.max_batch_size == 8

        input = config.add_input(
            name="test_input", shape=(None, 8), dtype="float32"
        )
        assert len(config.input) == 1
        assert config.input[0] == input
        assert config.input.name == "test_input"

        instance_group = config.add_instance_group(kind="gpu", gpus=2, count=4)
        assert len(config.instance_group) == 1
        assert instance_group._instance_group == config.instance_group[0]
        assert config.instance_group[0].kind == ModelInstanceGroup.KIND_GPU
        assert instance_group.gpus == [0, 1]
        assert instance_group.count == 4

        config.instance_groups[0].count = 6
        assert config.instance_group[0].count == 6
        assert config.instance_group[0].kind == ModelInstanceGroup.KIND_GPU
        assert instance_group.gpus == [0, 1]

        config.write()
        new_config = DummyModel.fs.read_config(
            DummyModel.fs.join(config.name, "config.pbtxt")
        )
        assert new_config == config
    except Exception:
        DummyModel.fs.delete()
        raise
