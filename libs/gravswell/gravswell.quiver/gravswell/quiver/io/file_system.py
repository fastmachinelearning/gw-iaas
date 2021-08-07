import abc
import typing
from dataclasses import dataclass

from google.protobuf import text_format
from tritonclient.grpc.model_config_pb2 import ModelConfig

_IO_TYPE = typing.Union[str, bytes]


@dataclass
class FileSystem(metaclass=abc.ABCMeta):
    root: str

    @abc.abstractmethod
    def soft_makedirs(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def delete(self, path: str):
        pass

    @abc.abstractmethod
    def read(self, path: str) -> _IO_TYPE:
        pass

    def read_config(self, path: str) -> ModelConfig:
        config = ModelConfig()
        config_txt = self.read(path)
        text_format.Merge(config_txt, config)
        return config

    @abc.abstractmethod
    def write(self, obj: _IO_TYPE, path: str) -> None:
        pass

    def write_config(self, config: ModelConfig, path: str) -> None:
        self.write(str(config), path)
