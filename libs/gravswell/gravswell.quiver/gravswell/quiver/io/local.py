import glob
import os
import shutil

from gravswell.quiver.io.file_system import _IO_TYPE, FileSystem


class LocalFileSystem(FileSystem):
    def soft_makedirs(self, path: str):
        # TODO: start using exists_ok kwargs once
        # we know we have the appropriate version
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return False

    def delete(self, path: str):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            paths = glob.glob(path)
            if len(paths) == 0:
                raise ValueError(
                    f"Could not find path or paths matching {path}"
                )
            for path in paths:
                self.delete(path)

    def read(self, path: str, mode: str = "r"):
        with open(path, mode) as f:
            return f.read()

    def write(self, obj: _IO_TYPE, path: str) -> None:
        if isinstance(obj, str):
            mode = "w"
        elif isinstance(obj, bytes):
            mode = "wb"
        else:
            raise TypeError(
                "Expected object to be of type "
                "str or bytes, found type {}".format(type(obj))
            )

        with open(path, mode) as f:
            f.write(obj)
