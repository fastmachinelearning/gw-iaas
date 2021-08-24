import re
import subprocess
import time
from contextlib import contextmanager
from threading import Event, Thread
from typing import Optional, Sequence

from rich.progress import BarColumn, Progress, TimeElapsedColumn


class GPUUtilDisplay(Thread):
    def __init__(self, progress, gpu_ids):
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

    def stop(self):
        self._stop_event.set()

    def run(self):
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
def get_progbar(gpu_ids: Optional[Sequence[int]] = None):
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
