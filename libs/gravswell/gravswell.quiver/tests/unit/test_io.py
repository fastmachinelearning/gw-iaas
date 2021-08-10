import os
import shutil

import pytest
from google.api_core.exceptions import NotFound

from gravswell.quiver.io import GCSFileSystem, LocalFileSystem


def run_file_manipulations(fs):
    fnames = [
        fs.join("test", "123", "test.txt"),
        fs.join("test", "456", "test.txt"),
        fs.join("test", "test.txt"),
        fs.join("test", "123", "test.csv"),
    ]

    for f in fnames:
        fs.write("testing", f)

    assert fs.list("test") == ["123", "456", "test.txt"]
    assert fs.glob(fs.join("test", "*.txt")) == fnames[:-1]

    for f in fnames:
        assert fs.read(f) == "testing"

    fs.remove(fnames[-1])
    with pytest.raises(FileNotFoundError):
        fs.read(fnames[-1])


def test_local_filesytem():
    dirname = "gravswell-quiver-test"
    fs = LocalFileSystem(dirname)
    assert os.path.isdir(dirname)

    try:
        assert fs.join("testing", "123") == os.path.join("test", "123")

        run_file_manipulations(fs)
        fs.delete()
        assert not os.path.isdir(dirname)
    except Exception:
        shutil.rmtree(dirname)


def test_gcs_filesystem():
    bucket_name = "gs://gravswell-quiver-test"
    with pytest.raises(ValueError):
        fs = GCSFileSystem(bucket_name.replace("gs://", ""))

    fs = GCSFileSystem(bucket_name)
    try:
        assert fs.soft_makedirs("")
        assert fs.join("testing", "123") == "testing/123"

        run_file_manipulations(fs)

        fs.delete()
        with pytest.raises(NotFound):
            fs.client.get_bucket(bucket_name)
    except Exception:
        fs.bucket.delete(force=True)
        raise
