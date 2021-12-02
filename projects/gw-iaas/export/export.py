import logging
import sys
from typing import Optional, Sequence, Union

import torch
from deepclean_prod.nn.net import DeepClean
from mldet.net import Net as BBHNet

from hermes import quiver as qv
from hermes.typeo import typeo


class OnlineDeepClean(torch.nn.Module):
    def __init__(self, deepclean, step_size):
        super().__init__()

        self.deepclean = deepclean
        self.step_size = step_size

    def __call__(self, x):
        return self.deepclean(x)[:, -self.step_size :]


class PostProcessor(torch.nn.Module):
    def forward(self, strain, noise_h, noise_l):
        # TODO: needs to add:
        #    - filtering
        #    - de-centering
        #    - any preprocessing for bbh
        noise = torch.stack([noise_h, noise_l], dim=1)
        return strain - noise


def parse_channels(channels):
    if isinstance(channels, str) or len(channels) == 1:
        if not isinstance(channels, str):
            channels = channels[0]
        with open(channels, "r") as f:
            channels = [i for i in f.read().splitlines() if i]
    return channels


def export_deepclean(
    repo: qv.ModelRepository,
    num_channels: int,
    kernel_size: int,
    step_size: Optional[int] = None,
    instances: Optional[int] = None,
    postfix: Optional[str] = None,
    weights: Optional[str] = None,
    platform: qv.Platform = qv.Platform.ONNX,
) -> qv.Model:
    deepclean = DeepClean(num_channels)
    if weights is not None:
        deepclean.load_state_dict(torch.load(weights))

    if step_size is not None:
        deepclean = OnlineDeepClean(deepclean, step_size)
    deepclean.eval()

    name = "deepclean" + (postfix or "")
    model = repo.add(name, platform=platform)

    if instances is not None:
        model.config.add_instance_group(count=instances)

    model.export_version(
        deepclean,
        input_shapes={"witness": (1, num_channels, kernel_size)},
        output_names=["noise"],
    )
    return model


def export_bbh(
    repo: qv.ModelRepository,
    kernel_size: int,
    instances: Optional[int] = None,
    postfix: Optional[str] = None,
    weights: Optional[str] = None,
    platform: qv.Platform = qv.Platform.ONNX,
):
    # TODO: load these from a config somewhere?
    bbh_params = {
        "filters": (3, 3, 3),
        "kernels": (8, 16, 32),
        "pooling": (4, 4, 4, 4),
        "dilations": (1, 1, 1, 1),
        "pooling_type": "max",
        "pooling_first": True,
        "bn": True,
        "linear": (64, 32),
        "dropout": 0.5,
        "weight_init": None,
        "bias_init": None,
    }
    bbh = BBHNet((2, kernel_size), bbh_params)
    if weights is not None:
        bbh.load_state_dict(torch.load(weights))
    bbh.eval()

    name = "bbh" + (postfix or "")
    model = repo.add(name, platform=platform)

    if instances is not None:
        model.config.add_instance_group(count=instances)
    model.export_version(
        bbh,
        input_shapes={"strain": (None, 2, kernel_size)},
        output_names=["prob"],
    )
    return model


def deepclean(
    repo_dir: str,
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    channels: Union[str, Sequence[str]],
    streams_per_gpu: int = 1,
    instances: Optional[int] = None,
    max_latency: Optional[float] = None,
    postfix: Optional[str] = None,
    weights: Optional[str] = None,
    platform: qv.Platform = qv.Platform.ONNX,
) -> None:
    repo = qv.ModelRepository(repo_dir)
    channels = parse_channels(channels)

    if max_latency is not None:
        num_updates = max_latency // stride_length
        step_size = int(stride_length * num_updates * sample_rate)
    else:
        step_size = int(stride_length * sample_rate)

    model = export_deepclean(
        repo,
        num_channels=len(channels) - 1,
        kernel_size=int(kernel_length * sample_rate),
        step_size=step_size,
        instances=instances,
        postfix=postfix,
        weights=weights,
        platform=platform,
    )

    name = "deepclean-stream" + (postfix or "")
    ensemble = repo.add(name, platform=qv.Platform.ENSEMBLE)

    ensemble.add_streaming_inputs(
        inputs=model.inputs["witness"],
        stream_size=int(stride_length * sample_rate),
        name="snapshotter" + (postfix or ""),
        streams_per_gpu=streams_per_gpu,
    )

    if max_latency is not None:
        ensemble.add_streaming_output(
            model.outputs["noise"],
            int(stride_length * sample_rate),
            num_updates,
            streams_per_gpu=streams_per_gpu,
        )
    else:
        ensemble.add_output(model.outputs["noise"])
    ensemble.export_version(None)


def end_to_end(
    repo_dir: str,
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    hanford_channels: Union[str, Sequence[str]],
    livingston_channels: Union[str, Sequence[str]],
    hanford_instances: Optional[int] = None,
    livingston_instances: Optional[int] = None,
    bbh_instances: Optional[int] = None,
    hanford_weights: Optional[str] = None,
    livingston_weights: Optional[str] = None,
    bbh_weights: Optional[str] = None,
    streams_per_gpu: int = 1,
    postfix: Optional[str] = None,
    hanford_platform: qv.Platform = qv.Platform.ONNX,
    livingston_platform: qv.Platform = qv.Platform.ONNX,
    bbh_platform: qv.Platform = qv.Platform.ONNX,
) -> None:
    repo = qv.ModelRepository(repo_dir)
    kernel_size = int(kernel_length * sample_rate)

    hanford_channels = parse_channels(hanford_channels)
    hanford = export_deepclean(
        repo,
        num_channels=len(hanford_channels) - 1,
        kernel_size=kernel_size,
        step_size=None,
        instances=hanford_instances,
        postfix="-hanford" + (postfix or ""),
        weights=hanford_weights,
        platform=hanford_platform,
    )

    livingston_channels = parse_channels(livingston_channels)
    livingston = export_deepclean(
        repo,
        num_channels=len(livingston_channels) - 1,
        kernel_size=kernel_size,
        step_size=None,
        instances=livingston_instances,
        postfix="-livingston" + (postfix or ""),
        weights=livingston_weights,
        platform=livingston_platform,
    )

    bbh = export_bbh(
        repo,
        kernel_size=kernel_size,
        instances=bbh_instances,
        postfix=postfix,
        weights=bbh_weights,
        platform=bbh_platform,
    )

    postprocessor = repo.add("postprocessor", platform=qv.Platform.ONNX)
    postprocessor.export_version(
        PostProcessor(),
        input_shapes={
            "strain": (1, 2, kernel_size),
            "noise_h": (1, kernel_size),
            "noise_l": (1, kernel_size),
        },
        output_names=["cleaned"],
    )

    name = "end-to-end" + (postfix or "")
    ensemble = repo.add(name, platform=qv.Platform.ENSEMBLE)
    ensemble.add_streaming_inputs(
        inputs=[
            hanford.inputs["witness"],
            livingston.inputs["witness"],
            postprocessor.inputs["strain"],
        ],
        stream_size=int(stride_length * sample_rate),
        name="snapshotter" + (postfix or ""),
        streams_per_gpu=streams_per_gpu,
    )

    ensemble.pipe(hanford.outputs["noise"], postprocessor.inputs["noise_h"])
    ensemble.pipe(livingston.outputs["noise"], postprocessor.inputs["noise_l"])
    ensemble.pipe(postprocessor.outputs["cleaned"], bbh.inputs["strain"])

    ensemble.add_output(bbh.outputs["prob"])
    ensemble.export_version(None)


@typeo("Export", deepclean=deepclean, end_to_end=end_to_end)
def main(filename: Optional[str] = None, verbose: bool = False) -> None:
    if filename is None:
        kwargs = {"stream": sys.stdout}
    else:
        kwargs = {"file": filename}

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        **kwargs
    )


if __name__ == "__main__":
    main()
