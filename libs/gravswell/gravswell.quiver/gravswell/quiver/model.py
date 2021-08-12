from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from gravswell.quiver import ModelConfig, platforms

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from triton.model_config_pb2 import ModelEnsembling, types

    from gravswell.quiver import ModelRepository
    from gravswell.quiver.io.file_system import FileSystem


@dataclass
class ExposedTensor:
    """An input or output tensor to a model.

    Captures descriptive information about inputs
    and outputs of models to allow for easier
    piping between models in ensembles.

    Args:
        model:
            The `Model` to which this tensor belongs
        name:
            The name of this tensor
        shape:
            The shape of this tensor
    """

    model: "Model"
    name: str
    shape: types.SHAPE_TYPE


_TENSOR_TYPE = Union[str, ExposedTensor]


@dataclass
class Model:
    """An entry within a model repository

    Args:
        name:
            The name of the model
        repository:
            The model repository to which this model belongs
        platform:
            The backend platform used to execute inference
            for this model

    Attributes:
        config:
            The config associated with this model. If one
            already exists in the model repository at
            initialization, it is loaded in and verified
            against the initialization parameters. Otherwise,
            one is initialized using only the `Model`
            initialization params. Information about model
            inputs and outputs is inferred at the time of
            the first version export, or can be specified
            directly onto the config.
    """

    name: str
    repository: ModelRepository
    platform: platforms.Platform

    def __new__(
        cls,
        name: str,
        repository: ModelRepository,
        platform: platforms.Platform,
    ):
        if isinstance(platform, platforms.ENSEMBLE):
            cls = EnsembleModel
        return super().__new__(cls)

    def __post_init__(self):
        self.repository.fs.soft_makedirs(self.name)
        self.config = ModelConfig(self)

    @property
    def fs(self) -> FileSystem:
        """The `FileSystem` leveraged by the model's repository"""

        return self.repository.fs

    @property
    def versions(self) -> list[int]:
        """The existing versions of this model in the repository"""

        # TODO: implement a `walk` method on the
        # filesystems, that way cloud based ones
        # don't have to do two object listings
        # here: one for list and one implicitly
        # inside isdir
        versions = []
        for f in self.fs.list(self.name):
            if self.fs.isdir(f):
                try:
                    version = int(f)
                except ValueError:
                    continue
                versions.append(version)
        return versions

    @property
    def inputs(self) -> dict[str, ExposedTensor]:
        """The inputs exposed by this model

        Represented by a dictionary mapping from the name
        of each input to the corresponding `ExposedTensor`
        """
        inputs = {}
        for input in self.config.input:
            shape = tuple(x if x != -1 else None for x in input.dims)
            inputs[input.name] = ExposedTensor(self, input.name, shape)
        return inputs

    @property
    def outputs(self) -> dict[str, ExposedTensor]:
        """The outputs exposed by this model

        Represented by a dictionary mapping from the name
        of each output to the corresponding `ExposedTensor`
        """
        outputs = {}
        for output in self.config.output:
            shape = tuple(x if x != -1 else None for x in output.dims)
            outputs[output.name] = types.ExposedTensor(
                self, output.name, shape
            )
        return outputs

    def export_version(
        self,
        model_fn: Union[Callable, "Model"],
        version: Optional[int] = None,
        input_shapes: Optional[dict[str, types.SHAPE_TYPE]] = None,
        output_names: Optional[Sequence[str]] = None,
        verbose: int = 0,
        **kwargs,
    ) -> str:
        """Export a version of this model to the repository

        Exports a model represented by `model_fn` to the
        model repository at the specified `version`.
        """
        # default version will be the latest
        version = version or len(self.versions) + 1

        try:
            output_dir = self.fs.join(self.name, str(version))

            # use boolean returned by soft_makedirs to
            # make sure we remove any directory we created
            # for this version if the export fails
            do_remove = self.fs.soft_makedirs(output_dir)

            # perform the export
            return self.platform.export(
                model_fn,
                version,
                input_shapes=input_shapes,
                output_names=output_names,
                verbose=verbose,
                **kwargs,
            )
        except Exception:
            # if anything goes wrong and we created a
            # directory above, make sure to get rid
            # of it before raising
            if do_remove:
                self.fs.remove(output_dir)
            raise


class EnsembleModel(Model):
    @property
    def models(self):
        return [
            self.repository.models[step.model_name]
            for step in self._config.ensemble_scheduling.step
        ]

    def _find_tensor(
        self,
        tensor: types.TENSOR_TYPE,
        exposed_type: types.EXPOSED_TYPE,
        version: Optional[int] = None,
    ) -> Union[ExposedTensor, "ModelEnsembling.Step"]:
        if isinstance(tensor, str):
            for model in self._models:
                tensors = getattr(model, exposed_type + "s")
                try:
                    tensor = tensors[tensor]
                except KeyError:
                    continue
                break
            else:
                raise ValueError(
                    f"Coludn't find model with input {input} "
                    "in model repository."
                )

        for step in self.config.ensemble_scheduling.step:
            if step.model_name == tensor.model.name:
                break
        else:
            if tensor.model not in self._models:
                raise ValueError(
                    f"Trying to add model {tensor.model.name} to "
                    "ensemble that doesn't exist in repo."
                )
            self.config.add_step(tensor.model, version=version)
        return tensor

    def _update_step_map(
        self,
        model_name: str,
        key: str,
        value: str,
        exposed_type: types.EXPOSED_TYPE,
    ):
        for step in self.config.ensemble_scheduling.step:
            if step.model_name == model_name:
                step_map = getattr(step, exposed_type + "_map")
                step_map[key] = value

    def add_input(
        self,
        input: _TENSOR_TYPE,
        version: Optional[int] = None,
        name: Optional[str] = None,
    ) -> ExposedTensor:
        input = self._find_tensor(input, "input", version)
        name = name or input.name
        if input.name not in self.inputs:
            self.config.add_input(
                name,
                input.shape,
                dtype="float32",  # TODO: dynamic dtype mapping
            )
        self._update_step_map(input.model.name, input.name, name, "input")
        return self.inputs[name]

    def add_output(
        self,
        output: _TENSOR_TYPE,
        version: Optional[int] = None,
        name: Optional[str] = None,
    ) -> ExposedTensor:
        output = self._find_tensor(output, "output", version)
        name = name or output.name
        if output.name not in self.outputs:
            self.config.add_output(
                name,
                output.shape,
                dtype="float32",  # TODO: dynamic dtype mapping
            )
        self._update_output_map(output.model.name, output.name, name)
        return self.outputs[name]

    def add_streaming_inputs(
        self,
        inputs: Union[Sequence[_TENSOR_TYPE], _TENSOR_TYPE],
        stream_size: int,
        name: Optional[str] = None,
        streams_per_gpu: int = 1,
    ):
        if not isinstance(inputs, Sequence):
            inputs = [inputs]

        tensors = []
        for input in inputs:
            tensor = self._find_tensor(input, "input")
            tensors.append(tensor)

        try:
            from exportlib.stream import make_streaming_input_model
        except ImportError as e:
            if "tensorflow" in str(e):
                raise RuntimeError(
                    "Unable to leverage streaming input, "
                    "must install TensorFlow first"
                )
        streaming_model = make_streaming_input_model(
            self.repository, tensors, stream_size, name, streams_per_gpu
        )

        self.add_input(streaming_model.inputs["stream"])

        metadata = []
        for tensor, output in zip(tensors, streaming_model.config.output):
            self.pipe(streaming_model.outputs[output.name], tensor)
            metadata.append("{}/{}".format(tensor.model.name, tensor.name))

        self.config.parameters["states"].string_value = ",".join(metadata)

    def pipe(
        self,
        input: Union[str, ExposedTensor],
        output: Union[str, ExposedTensor],
        name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> None:
        input = self._find_tensor(input, "output")
        output = self._find_tensor(output, "input", version)

        try:
            for step in self.config.ensemble_scheduling.step:
                if step.model_name == input.model.name:
                    break
            current_key = step.output_map[input.name]
            if current_key == "":
                raise KeyError
        except KeyError:
            name = name or input.name
            self._update_step_map(input.model.name, input.name, name, "output")
        else:
            if name is not None and current_key != name:
                raise ValueError(
                    f"Output {input.name} from {input.model.name} "
                    f"already using key {current_key}, couldn't "
                    f"use provided key {name}"
                )
            name = current_key

        try:
            for step in self.config.ensemble_scheduling.step:
                if step.model_name == output.model.name:
                    break
            current_key = step.input_map[output.name]
            if current_key == "":
                raise KeyError
        except KeyError:
            self._update_step_map(
                output.model.name, output.name, name, "output"
            )
        else:
            if current_key != name:
                raise ValueError(
                    f"Input {output.name} to {output.model.name} "
                    f"already receiving input from {current_key}, "
                    f"can't pipe input {name}"
                )
