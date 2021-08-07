import os

from tritonclient.grpc.model_config_pb2 import ModelConfig

from gravswell.quiver.io import GCSFileSystem, LocalFileSystem


def test_local_filesytem():
    fs = LocalFileSystem(".")

    model_dir = os.path.join("repo", "deepclean")
    assert fs.soft_makedirs(model_dir)
    assert os.path.exists(model_dir)

    config = ModelConfig(name="test")
    config_path = os.path.join(model_dir, "config.pbtxt")
    fs.write_config(config, config_path)
    assert os.path.exists(config_path)

    config = fs.read_config(config_path)
    assert config.name == "test"

    fs.delete("repo")


def test_gcs_filesystem():
    fs = GCSFileSystem("gravswell-quiver-test")

    config = ModelConfig(name="test")
    config_path = "repo/deepclean/config.pbtxt"
    fs.write_config(config, config_path)
    assert fs.bucket.get_blob(config_path) is not None

    config = fs.read_config(config_path)
    assert config.name == "test"

    fs.delete("repo/deepclean/")
    fs.bucket.delete(force=True)
