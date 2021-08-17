import os

from gravswell.quiver import Model, Platform
from gravswell.quiver.io import LocalFileSystem


class DummyRepo:
    def __enter__(self):
        self.fs = LocalFileSystem("gravswell-quiver-test")
        return self

    def __exit__(self, *exc_args):
        self.fs.delete()


def test_model():
    with DummyRepo() as repo:
        model = Model("test", repo, platform=Platform.ONNX)

        assert os.path.exists(os.path.join("gravswell-quiver-test", "test"))
        assert len(model.versions) == 0
