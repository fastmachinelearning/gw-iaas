import typing
from dataclasses import dataclass

try:
    from google.api_core.exceptions import Forbidden, NotFound
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage

    _has_google_libs = True
except ImportError:
    _has_google_libs = False

from gravswell.quiver.io.file_system import _IO_TYPE, FileSystem


@dataclass
class GCSFileSystem(FileSystem):
    credentials: typing.Optional[str]

    def __post_init__(self):
        if not _has_google_libs:
            raise ImportError(
                "Must install google-cloud-storage to use GCSFileSystem"
            )

        if self.credentials is not None:
            self.client = storage.Client.from_service_account_json(
                self.credentials
            )
        else:
            try:
                self.client = storage.Client()
            except DefaultCredentialsError:
                raise ValueError(
                    "Must specify service account json file "
                    "via the `GOOGLE_APPLICATION_CREDENTIALS` "
                    "environment variable to use a GCSFileSystem"
                )

        bucket_name = self.root.strip("gs://").split("/")[0]
        try:
            self.bucket = self.client.get_bucket(bucket_name)
        except NotFound:
            self.bucket = self.client.create_bucket(bucket_name)
        except Forbidden:
            raise ValueError(
                "Provided credentials are unable to access "
                f"GCS bucket with name {bucket_name}"
            )

        try:
            self.prefix = self.root.strip("gs://").split("/", maxsplit=1)[1]
        except IndexError:
            self.prefix = None

    def soft_makedirs(self, path: str):
        return True

    def delete(self, path: str):
        postfix = None
        if "*" in path:
            splits = path.split("*")
            if len(splits) > 2:
                raise ValueError(f"Could not parse path {path}")
            path, postfix = splits

        for blob in self.bucket.list_blobs(prefix=path):
            if postfix is None:
                blob.delete()
            elif blob.name.endswith("postfix"):
                blob.delete()

    def read(self, path: str, mode: str = "r"):
        blob = self.bucket.get_blob(path)
        if blob is None:
            raise FileNotFoundError(path)

        content = blob.download_as_bytes()
        if mode == "r":
            content = content.decode()
        return content

    def write(self, obj: _IO_TYPE, path: str) -> None:
        blob = self.bucket.get_blob(path)
        if blob is not None:
            blob.delete()
        blob = self.bucket.blob(path)

        if isinstance(obj, str):
            content_type = "text/plain"
        elif isinstance(obj, bytes):
            content_type = "application/octet-stream"
        else:
            raise TypeError(
                "Expected object to be of type "
                "str or bytes, found type {}".format(type(obj))
            )

        blob.upload_from_string(obj, content_type=content_type)
