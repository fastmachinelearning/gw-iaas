from dataclasses import dataclass
from typing import TYPE_CHECKING

from tritonclient.grpc import model_config_pb2 as model_config

from gravswell.quiver.platforms import Ensemble

if TYPE_CHECKING:
    from abc.collections import Sequence
    from typing import Literal, Optional, Union

    from gravswell.quiver import Model
    from gravswell.quiver.types import SHAPE_TYPE


KIND_TYPE = Literal["auto", "cpu", "gpu"]
GPUS_TYPE = Union[int, Sequence[int], None]


def _normalize_kind(kind: KIND_TYPE) -> int:
    try:
        return model_config.ModelInstanceGroup.Kind.Value(
            "KIND_{}".format(kind.upper())
        )
    except ValueError:
        raise ValueError(
            f"Could not understand instance group kind {kind}, "
            "must be one of auto, gpu, cpu"
        )


def _normalize_gpus(gpus: GPUS_TYPE) -> list[int]:
    """Normalize a specified number of gpus to the protobuf syntax

    Passing a single integer will be interpreted as a
    range of GPU indices.
    """
    if gpus is None:
        return []
    elif isinstance(gpus, int):
        if gpus < 1:
            raise ValueError(f"Invalid number of gpus specified {gpus}")
        return [i for i in range(gpus)]
    return gpus


@dataclass
class InstanceGroup:
    _instance_group: model_config.ModelInstanceGroup

    def update(self, **kwargs) -> None:
        new_instance_group = model_config.ModelInstanceGroup(**kwargs)
        self._instance_group.MergeFrom(new_instance_group)

    @property
    def kind(self) -> Optional[str]:
        kind = self._instance_group.kind
        if kind == model_config.ModelInstanceGroup.KIND_CPU:
            return "cpu"
        elif kind == model_config.ModelInstanceGroup.KIND_GPU:
            return "gpu"
        elif kind == model_config.ModelInstanceGroup.KIND_AUTO:
            return "auto"
        else:
            # TODO: how do we want to suppor the
            # other kinds, and what to do until then?
            # Raise an error, return None?
            return None

    @kind.setter
    def kind(self, kind: KIND_TYPE) -> None:
        kind = _normalize_kind(kind)
        self.update(kind=kind)

    @property
    def gpus(self) -> list[int]:
        return self._instance_group.gpus

    @gpus.setter
    def gpus(self, gpus: GPUS_TYPE):
        gpus = _normalize_gpus(gpus)
        self.update(gpus=gpus)

    @property
    def count(self) -> int:
        return self._instance_group.count

    @count.setter
    def count(self, count: int) -> None:
        self.update(count=count)

    def __repr__(self) -> str:
        return self._instance_group.__repr__()

    def __str__(self) -> str:
        return str(self._instance_group)


def _add_exposed_tensor(f):
    """
    Decorator for adding input/output adding methods
    to the config class. Doing it this way in order to simplify
    things like syntax updates and building the data type map
    """
    exposed_type = f.__name__.split("_")[1]
    output_type = getattr(model_config, "Model" + exposed_type.title())

    def wrapper(
        obj: "ModelConfig",
        name: str,
        shape: SHAPE_TYPE,
        dtype: str = Literal["float32", "int64"],
        **kwargs,  # including kwargs for reshaping later or something
    ) -> output_type:
        """Add an {exposed} tensor to the config

        Appends an additional entry to `ModelConfig.{exposed}`
        with the specified keyword arguments.

        Args:
            name:
                The name of the {exposed}
            shape:
                The shape of the {exposed}, with `None`
                representing variable-length axes
            dtype:
                The datatype of the {exposed}
        """.format(
            exposed=exposed_type
        )

        # TODO: Handle datatypes more robustly
        if dtype == "float32":
            dtype = model_config.DataType.TYPE_FP32
        elif dtype == "int64":
            dtype = model_config.DataType.TYPE_INT64
        else:
            raise ValueError(f"Unknown datatype {dtype}")

        shape = (x or -1 for x in shape)
        exposed_obj = output_type(
            name=name,
            dims=shape,
            data_type=dtype,
        )

        current_exposed = getattr(obj._config, exposed_type)
        current_exposed.append(exposed_obj)
        f(exposed_obj, **kwargs)
        return exposed_obj

    wrapper.__name__ = f.__name__
    return wrapper


class ModelConfig:
    """Wrapper around the `tritonclient.grpc.model_config_pb2.ModelConfig`.

    Args:
        model:
            The `Model` object to which this config belongs
        **kwargs:
            Any additional keyword arguments
            with which to initialize the config
    """

    def __new__(cls, model: Model, **kwargs) -> "ModelConfig":
        if isinstance(model.platform, Ensemble):
            cls = EnsembleConfig

        obj = super().__new__(cls)
        obj.__init__(model, **kwargs)
        return obj

    def __init__(self, model: Model, **kwargs) -> None:
        self.model = model

        # make sure no kwargs were passed that
        # might override the values grabbed
        # from the Model itself
        if "name" in kwargs:
            raise ValueError(
                "Cannot pass 'name' as an argument to ModelConfig"
            )
        elif "platform" in kwargs:
            raise ValueError(
                "Cannot pass 'platform' as an argument to ModelConfig"
            )

        try:
            # try to read an existing config if it exists
            config = self.fs.read_config(
                self.fs.join(model.name, "config.pbtxt")
            )

            # ensure that the name in the config
            # matches the name passed from the model
            if config.name != model.name:
                raise ValueError(
                    "Name in existing config {} "
                    "doesn't match model name {}".format(
                        config.name, model.name
                    )
                )

            # do the same for the platform
            if config.platform != model.platform.convention.name:
                raise ValueError(
                    "Platform in existing config {} "
                    "doesn't match model platform {}".format(
                        config.platform, model.platform.convention.name
                    )
                )

            # add in any kwargs passed to overwrite
            # their existing value in the config
            kwargs_config = model_config.ModelConfig(**kwargs)
            config.MergeFrom(kwargs_config)

        except FileNotFoundError:
            # create a new config if one doesn't
            # already exist
            config = model_config.ModelConfig(
                name=model.name,
                platform=model.platform.convention.name,
                **kwargs,
            )

        # add it as an attribute
        self._config = config

    def __getattr__(self, name):
        """
        Override of `__getattr__` to look for
        attributes directly on the `_config` object
        if they don't exist on this object, e.g.
        `ModelConfig.input` will return the `ModelInput`
        message on the underlying `ModelConfig._config`
        """
        try:
            return self._config.__getattribute__(name)
        except AttributeError as e:
            raise AttributeError from e

    def write(self):
        """
        Write out the protobuf config to the model's
        folder in the model repository
        """
        path = self.model.fs.join(self.model.name, "config.pbtxt")
        self.model.fs.write_config(self._config, path)

    @_add_exposed_tensor
    def add_input(input: model_config.ModelInput, **kwargs):
        """
        add an input
        """
        return

    @_add_exposed_tensor
    def add_output(output: model_config.ModelOutput, **kwargs):
        """
        add an output
        """
        return

    @property
    def instance_groups(self):
        return [InstanceGroup(g) for g in self._config.instance_group]

    def add_instance_group(
        self,
        kind: KIND_TYPE = "gpu",
        gpus: GPUS_TYPE = None,
        count: int = 1,
    ) -> InstanceGroup:
        # first add a blank initialized
        # instance group to the config
        new_instance_group = model_config.ModelInstanceGroup()
        self.instance_group.append(new_instance_group)

        # wrap it in an `InstanceGroup` object
        # to let it handle the translation
        # of the arguments to the expected
        # protobuf syntax
        instance_group = InstanceGroup(new_instance_group)
        instance_group.kind = kind
        instance_group.gpus = gpus
        instance_group.count = count

        # return the compiled InstanceGroup wrapper
        return instance_group

    def __repr__(self):
        return self._config.__repr__()

    def __str__(self):
        return str(self._config)


class EnsembleConfig(ModelConfig):
    def add_step(self, model: "Model", version: Optional[int] = None):
        version = version or -1
        step = model_config.ModelEnsembling.Step(
            model_name=model.name, model_version=version
        )
        self._config.ensemble_scheduling.step.append(step)
        return step
