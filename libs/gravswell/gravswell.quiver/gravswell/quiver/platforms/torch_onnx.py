import inspect
from collections import OrderedDict
from dataclasses import dataclass

try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

from gravswell.quiver.platforms import Convention, Platform


@dataclass
class TorchOnnx(Platform):
    def _validate_platform(self):
        if not _has_torch:
            raise ImportError(
                "Must have torch installed to use " "TorchOnnx export platform"
            )

    def _get_tensor(self, shape):
        tensor_shape = []
        for dim in shape:
            if dim == -1:
                dim = self.model_config.max_batch_size or 1
            tensor_shape.append(dim)

        # TODO: this will not always be safe, for
        # example if your inputs need to be in a
        # certain range or are e.g. categorical
        # Should we expose an optional `input_tensors`
        # argument? We could accept framework tensors
        # at the `input_shapes` arg of `Platform.export`
        # and pass them along if they were provided?
        return torch.random.randn(*shape)

    def _get_output_shapes(self, model_fn, output_names):
        # now that we know we have inputs added to our
        # model config, use that config to generate
        # framework tensors that we'll feed through
        # the network to inspect the output
        input_tensors = OrderedDict()
        for input in self.model.config.input:
            # generate an input array of random data
            input_tensors[input.name] = self._get_tensor(input.dim)

        # use function signature from module.forward
        # to figure out in which order to pass input
        # tensors to the model_fn
        if isinstance(model_fn, torch.nn.Module):
            signature = inspect(model_fn.forward)
        else:
            signature = inspect(model_fn)
        parameters = OrderedDict(signature.parameters)

        # make sure the number of inputs to
        # the model_fn matches the number of
        # specified input tensors
        if len(parameters) != len(input_tensors):
            raise ValueError(
                "Attempted to export a model function "
                "for model {} which accepts {} inputs, "
                "but {} are expected".format(
                    self.model.name, len(parameters), len(input_tensors)
                )
            )

        if len(parameters) == 1:
            # if we have simple 1 -> 1 mapping,
            # don't impose requirements on matching
            # names and just use the given input
            # tensor as-is
            input_names = list(input_tensors)
        else:
            input_names = list(parameters)

        try:
            # first try to grab the input tensors
            # by name from the input_tensors
            # dictionary, using the names of the
            # arguments to the model_fn
            input_tensors = [input_tensors[name] for name in input_names]
        except KeyError:
            # at least one of the argument names didn't
            # match an input name in the config, so use
            # the ordering in the config as the backup
            input_tensors = list(input_tensors.values())

        # run the model_fn on the inputs in order
        # to get the model outputs
        outputs = model_fn(*input_tensors)

        # if we only have one output, wrap it
        # in a list to keep things generic
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        # grab the output shapes
        shapes = [tuple(x.shape) for x in outputs]

        # if any of the inputs have a variable
        # length batch dimension, then each output
        # should have a variable length batch
        # dimension too
        if any([x.dims[0] == -1 for x in self.model.config.input]):
            shapes = [(None,) + s[1:] for s in shapes]

        # if we provided names for the outputs,
        # return them as a dict for validation
        # against the config
        if output_names is not None:
            shapes = {name: shape for name, shape in zip(output_names, shapes)}
        return shapes

    @property
    def convention(self):
        return Convention.ONNX

    def _do_export(self, model_fn, export_obj, verbose=0):
        inputs, dynamic_axes = [], {}
        for input in self.model.config.input:
            if input.dim[0] == -1:
                dynamic_axes[input.name] = {0: "batch"}
            inputs.append(self._get_tensor(input.dim))

        if len(dynamic_axes) > 0:
            for output in self.model.config.output:
                dynamic_axes[output.name] = {0: "batch"}

        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs = tuple(inputs)

        torch.onnx.export(
            model_fn,
            inputs,
            export_obj,
            input_names=[x.name for x in self.model.config.input],
            output_names=[x.name for x in self.model.config.output],
            dynamic_axes=dynamic_axes or None,
        )
        return export_obj
