import os
import re
import subprocess
import time
from threading import Event, Thread
from typing import Callable, Sequence, Union

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TimeElapsedColumn


class GpuUtilProgress(Progress):
    """Progress bar subclass for measuring GPU utilization"""

    def __init__(
        self, gpu_ids: Union[int, Sequence[int], None] = None, *args
    ) -> None:
        if len(args) == 0:
            super().__init__(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
            )
        else:
            super().__init__(*args)

        # if we specified gpu ids to monitor, create
        # a progress bar for each of them that shows
        # their utilization level
        if gpu_ids is not None:
            # normalize a single integer gpu id
            # to a sequence for simplicity
            if isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]

            for gpu_id in gpu_ids:
                self.add_task(
                    f"[green]GPU {gpu_id} utilization", total=100, start=False
                )

        # use an event to stop the internal loop
        self._stop_event, self._gpu_monitor = None, None

    @property
    def gpu_tasks(self) -> dict:
        """The subset of tasks that represent GPUs to monitor"""

        return {
            task_id: task
            for task_id, task in self._tasks.items()
            if re.fullmatch("\[green\]GPU [0-9] utilization", task.description)
        }

    def __enter__(self):
        # if we have any GPUs to monitor, launch a thread
        # to monitor them with a stop condition via an Event
        gpu_tasks = self.gpu_tasks
        if len(gpu_tasks) > 0:
            # start the gpu tasks first
            for task_id in gpu_tasks:
                self.start_task(task_id)

            self._stop_event = Event()
            self._gpu_monitor = Thread(target=self.monitor_utilization)
            self._gpu_monitor.start()

        # do whatever the usual Progress
        # context entering business is
        return super().__enter__()

    def __exit__(self, *exc_args):
        # if we were monitoring any GPUs, stop monitoring now
        if self._gpu_monitor is not None:
            # set the event to kill the loop
            self._stop_event.set()

            # wait for the thread to finish
            self._gpu_monitor.join()

        # reset these attributes
        self._stop_event, self._gpu_monitor = None, None

        # run the normal context exiting buisness
        super().__exit__(*exc_args)

    def monitor_utilization(self) -> None:
        # grab the GPU tasks up front just in case there are
        # a lot of other tasks so we don't have to be doing
        # a dict comprehension every loop iteration
        tasks = self.gpu_tasks

        while not self._stop_event.is_set():
            # get the output from nvidia-smi
            nv_smi = subprocess.run(
                "nvidia-smi", shell=True, capture_output=True
            ).stdout.decode()
            percentages = list(map(int, re.findall("[0-9]{1,3}(?=%)", nv_smi)))

            # parse information for each one of our GPUs
            for task_id, task in tasks.items():
                # get the GPU id from the task description
                gpu_id = int(re.search("[0-9]", task.description).group(0))

                # use this index to grab the corresponding utilization
                # and update the corresponding task with the delta
                delta = percentages[gpu_id] - task.completed
                self.update(task_id, advance=delta)

            # sleep to avoid having this thread overwhelm everything
            time.sleep(0.05)


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
    X: np.ndarray,
    model: torch.nn.Module,
    model_args: tuple,
    inference_fn: Callable,
    q: torch.multiprocessing.Queue,
    synchronizer: torch.multiprocessing.Value,
    jobs_per_gpu,
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
        X:
            The array containing _all_ data on which
            inference is desired
        q:
            The queue in which to place inference outputs
        synchronizer:
            A counter for synchronizing model inference
            across all the jobs, waiting until all
            models are created and have weights loaded
            into memory
        jobs_per_gpu:
            The number of jobs to execute per gpu
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
    model = model(*model_args).cuda(device_index)
    model.load_state_dict(torch.load(weights_file))

    # increment the synchronizer to indicate
    # that this model is ready
    synchronizer.value += 1

    # wait until _all_ child processes have built
    # their models _and_ the parent process has
    # given the okay to start (hence the +1)
    while synchronizer.value < (jobs_per_gpu * num_gpus) + 1:
        time.sleep(0.01)

    # do inference and put it in the q
    for y in inference_fn(model, x, device_index=device_index):
        q.put(y)

        
class ThrottledDataset(torch.utils.data.IterableDataset):
    def __init__(self, x, device_index=0):
        self.x = torch.from_numpy(x).cuda(device_index)

    def __iter__(self):
        self.it = iter(self.x)
        return self

    def __next__(self):
        x = next(self.it)
        time.sleep(0.001)
        return x


@torch.no_grad()
def do_some_throttled_inference(model, dataset, batch_size=8, device_index=0):
    # move the data to the GPU in bulk
    gpu_dataset = ThrottledDataset(dataset, device_index)
    for x in torch.utils.data.DataLoader(gpu_dataset, batch_size=batch_size):
        y = model(x)
        yield y.cpu().numpy()

        
class NoiseRemovalModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()

        output_size = input_size
        self.layers = torch.nn.ModuleList()
        for size in hidden_sizes:
            self.layers.append(torch.nn.Linear(input_size, size))
            self.layers.append(torch.nn.ReLU())
            input_size = size

        for size in hidden_sizes[-2::-1]:
            self.layers.append(torch.nn.Linear(input_size, size))
            self.layers.append(torch.nn.ReLU())
            input_size = size

        # add a sigmoid output layer
        self.layers.append(torch.nn.Linear(input_size, output_size))

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return x
 

def print_tree(d, n=0):
    root, ds, fs = next(os.walk(d))
    print("    "*n + os.path.basename(root) + "/")
    for d in ds:
        print_tree(os.path.join(root, d), n+1)

    for f in fs:
        print("    " * (n+1) + f)
