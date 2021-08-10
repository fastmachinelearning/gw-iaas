import typing
from dataclasses import dataclass

try:
    from google.api_core.exceptions import Forbidden, NotFound
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage

    _has_google_libs = True
except ImportError:
    _has_google_libs = False

from gravswell.quiver.io.exceptions import NoFilesFoundError
from gravswell.quiver.io.file_system import _IO_TYPE, FileSystem


@dataclass
class GCSFileSystem(FileSystem):
    credentials: typing.Optional[str] = None

    def __post_init__(self):
        # TODO: do we want to just have people
        # skip the `gs` portion then and just
        # pass the bucket name + prefix?
        if not self.root.startswith("gs://"):
            raise ValueError(f"GCSFileSystem root {self.root} is not valid")

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

        split_path = self.root.replace("gs://", "").split("/", maxsplit=1)
        try:
            bucket_name, prefix = split_path
            self.prefix = prefix.rstrip("/")
        except ValueError:
            bucket_name = split_path[0]
            self.prefix = ""

        try:
            # try to get the bucket if it already exists
            self.bucket = self.client.get_bucket(bucket_name)
        except NotFound:
            # if it doesn't exist, try to create it
            # note that we don't need to worry about
            # name collisions because this would have
            # raised `Forbidden` rather than `NotFound`
            self.bucket = self.client.create_bucket(bucket_name)
        except Forbidden:
            # bucket already exists but the given
            # credentials are insufficient to access it
            raise ValueError(
                "Provided credentials are unable to access "
                f"GCS bucket with name {bucket_name}"
            )

    def soft_makedirs(self, path: str):
        """
        Does nothing in the context of a GCS bucket
        where objects need to be created one
        by one
        """
        return True

    def join(self, *args) -> str:
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError(
                    "join() argument must be str, not {}".format(type(arg))
                )
        return "/".join(args)

    def list(self, path: typing.Optional[str] = None) -> typing.List[str]:
        if path is not None:
            if self.prefix:
                prefix = self.joins(self.prefix, path)
            else:
                prefix = path
        else:
            prefix = self.prefix

        if not prefix.endswith("/"):
            prefix += "/"

        fs, dirs = [], set()
        for blob in self.bucket.list_blobs(prefix=prefix):
            name = blob.name.replace(prefix, "")
            try:
                d, _ = name.split("/", maxsplit=1)
                dirs.add(d)
            except ValueError:
                fs.append(name)
        return sorted(list(dirs)) + sorted(fs)

    def glob(self, path: str):
        postfix = None
        prefix = self.prefix
        if "*" in path and path != "*":
            splits = path.split("*")
            if len(splits) > 2:
                raise ValueError(f"Could not parse path {path}")

            _prefix, postfix = splits
            if _prefix and prefix:
                prefix = f"{prefix}/{_prefix}"
            elif _prefix:
                prefix = _prefix

        names = []
        for blob in self.bucket.list_blobs(prefix=prefix):
            if postfix is None or blob.name.endswith(postfix):
                names.append(blob.name)
        return names

    def remove(self, path: str):
        names = self.glob(path)
        if len(names) == 0:
            raise NoFilesFoundError(path)

        for name in names:
            blob = self.bucket.get_blob(name)
            blob.delete()

    def delete(self):
        self.bucket.delete(force=True)

    def read(self, path: str, mode: str = "r"):
        if self.prefix:
            path = self.join(self.prefix, path)

        blob = self.bucket.get_blob(path)
        if blob is None:
            raise FileNotFoundError(path)

        content = blob.download_as_bytes()
        if mode == "r":
            content = content.decode()
        return content

    def write(self, obj: _IO_TYPE, path: str) -> None:
        if self.prefix:
            path = self.join(self.prefix, path)

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
