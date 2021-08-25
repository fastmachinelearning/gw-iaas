import re
import subprocess
import time
from contextlib import contextmanager
from threading import Event, Thread
from typing import Optional, Sequence

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TimeElapsedColumn


class GPUUtilDisplay(Thread):
    def __init__(self, progress: Progress, gpu_ids: Sequence[int]) -> None:
        self.progress = progress

        # for each gpu id, make a bar that shows the
        # utilization and keep track of the associated
        # task id and most recent utilization for
        # performing updates
        self.task_map = {}
        for gpu_id in gpu_ids:
            task_id = self.progress.add_task(
                f"[green]GPU {gpu_id} utilization", total=100
            )
            self.task_map[gpu_id] = (task_id, 0)

        # keep track of the average utility across
        # the entire run to record at the end once
        # the progress bar has completed
        self.average_utils = {gpu_id: 0 for gpu_id in gpu_ids}

        # use an event to stop the internal loop
        self._stop_event = Event()

        super().__init__()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        n = 0
        while not self._stop_event.is_set():
            # get the output from nvidia-smi
            nv_smi = subprocess.run(
                "nvidia-smi", shell=True, capture_output=True
            ).stdout.decode()
            percentages = list(map(int, re.findall("[0-9]{1,3}(?=%)", nv_smi)))

            # parse information for each one of our GPUs
            for gpu_id, (task_id, last_util) in self.task_map.items():
                # grab the utilization from the nvidia-smi output
                new_util = percentages[gpu_id]

                # compute the update and record the
                # new value in the task map
                delta = new_util - last_util
                self.progress.update(task_id, advance=delta)
                self.task_map[gpu_id] = (task_id, new_util)

                # keep track of our running average
                n += 1
                av_update = (new_util - self.average_utils[gpu_id]) / n
                self.average_utils[gpu_id] += av_update

        # sleep to avoid having this overwhelm everything
        time.sleep(0.05)

        # now that the task has been ended, update each
        # progress bar to be frozen with the average value
        # from the course of the run
        for gpu_id, (task_id, last_util) in self.task_map.items():
            delta = self.average_utils[gpu_id] - last_util
            self.progress.update(task_id, advance=delta)


@contextmanager
def get_progbar(gpu_ids: Optional[Sequence[int]] = None) -> Progress:
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    )

    if gpu_ids is not None:
        gpu_util = GPUUtilDisplay(progress, gpu_ids)
        gpu_util.start()

    with progress:
        yield progress

        if gpu_ids is not None:
            gpu_util.stop()
            gpu_util.join()


class MLP(torch.nn.Module):
    """Simple ReLU multi-layer perceptron model"""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()

        # build all the hidden layers, using the
        # last hidden size (or input size) as the
        # input size to this layer
        self.layers = torch.nn.ModuleList()
        for size in hidden_sizes:
            self.layers.append(torch.nn.Linear(input_size, size))
            self.layers.append(torch.nn.ReLU())
            input_size = size

        # add a sigmoid output layer
        self.layers.append(torch.nn.Linear(input_size, 1))
        self.layers.append(torch.nn.Sigmoid())

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@torch.no_grad()
def do_some_inference(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 8,
    device_index: int = 0,
) -> np.ndarray:
    """Do inference on the array `X` using the provided model

    Iterates through an array of data in chunks and
    performance model inference on it, yielding the model
    output at each iteration.

    Args:
        model:
            The torch module to use for inference
        X:
            The array to perform inference on, with the 0th
            axis representing the batch dimension
        batch_size:
            The size of chunks to iterate through X in
        device_index:
            The GPU index to use for inference

    Returns:
        The model output for each input batch
    """

    # put the array on the specified GPU device
    X = torch.from_numpy(X).cuda(device_index)

    # set up a batched dataset iterator
    dataset = torch.utils.data.TensorDataset(X)
    for [x] in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        # do inference and send the result back to the CPU
        y = model(x)
        yield y.cpu().numpy()


def parallel_inference_task(
    i: int,
    jobs_per_gpu: int,
    X: np.ndarray,
    q: torch.multiprocessing.Queue,
    starter: torch.multiprocessing.Value,
    num_gpus: int = 1,
    weights_file: str = "model.pt",
) -> None:
    """Perform inference on a subset of an input array

    Iterate through a subset of an array of data in
    chunks and perform model inference using an MLP
    with weights stored in the specified file. The
    subset of `X` that is used for inference as well
    as the GPU assigned for inference is specified by
    the index `i` and the total number of jobs
    `jobs_per_gpu * num_gpus`.

    Args:
        i:
            The job index for this inference process
        jobs_per_gpu:
            The number of jobs to execute per gpu
        X:
            The array containing _all_ data on which
            inference is desired
        q:
            The queue in which to place inference outputs
        starter:
            A counter for synchronizing model inference
            across all the jobs, waiting until all
            models are created and have weights loaded
            into memory
        num_gpus:
            The number of GPUs to leverage for inference
        weights_file:
            A file to load trained model weights in with
    """

    # lazily find the subset of X to work on
    # assign the device index based on the job index
    x = np.array_split(X, jobs_per_gpu * num_gpus)[i]
    device_index = i // jobs_per_gpu

    # build the model and load in its weights
    model = MLP(64, [256, 128, 64]).cuda(device_index)
    model.load_state_dict(torch.load(weights_file))

    # increment the starter to indicate that
    # this model is ready
    starter.value += 1

    # wait until _all_ child processes have built
    # their models _and_ the parent process has
    # given the okay to start (hence the +1)
    while starter.value < (jobs_per_gpu * num_gpus) + 1:
        time.sleep(0.01)

    # do inference and put it in the q
    for y in do_some_inference(model, x, device_index=device_index):
        q.put(y)
