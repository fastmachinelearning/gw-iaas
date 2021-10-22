import logging
import os
import pickle
import sys
from queue import Empty
from typing import TYPE_CHECKING, Optional

import numpy as np
from gwpy.timeseries import TimeSeries
from scipy import signal

from hermes.gwftools.gwftools import _parse_frame_name
from hermes.stillwater import PipelineProcess

if TYPE_CHECKING:
    from multiprocessing import Queue


def get_logger(filename: Optional[str] = None, verbose: bool = False):
    kwargs = {
        "level": logging.DEBUG if verbose else logging.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    if filename is not None:
        kwargs["file"] = filename
    else:
        kwargs["stream"] = sys.stdout
    logging.basicConfig(**kwargs)

    return logging.getLogger()


class Preprocessor:
    def __init__(self, preproc_pkl: str, sample_rate: float):
        with open(preproc_pkl, "rb") as f:
            params = pickle.load(f)

        low = params["filt_fl"][0] * 2 / sample_rate
        high = params["filt_fh"][0] * 2 / sample_rate
        self.sos = signal.butter(
            params["filt_order"], [low, high], btype="bandpass", output="sos"
        )

        self.mean = params["mean"]
        self.std = params["std"]

    def center(self, x):
        return (x - self.mean) / self.std

    def uncenter(self, x):
        return x * self.std + self.mean

    def filter(self, x, axis=-1):
        return signal.sosfiltfilt(self.sos, x, axis=axis)

    def __call__(self, x):
        return self.center(x)


class FrameWriter(PipelineProcess):
    def __init__(
        self,
        write_dir: str,
        channel_name: str,
        step_size: int,
        sample_rate: float,
        strain_q: "Queue",
        preprocessor: Optional[Preprocessor] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        self.write_dir = write_dir
        self.channel_name = channel_name
        self.step_size = step_size
        self.sample_rate = sample_rate
        self.strain_q = strain_q
        self.preprocessor = preprocessor

        self._strains = []
        self._noises = np.array([])
        self._covered_idx = np.array([])
        self._frame_idx = 0

    def get_package(self):
        # now get the next inferred noise estimate
        noise_prediction = super().get_package()

        # first see if we have any new strain data
        # to collect
        try:
            if len(self._noises) == 0:
                fname, strain = self.strain_q.get(True, 10)
            else:
                fname, strain = self.strain_q.get(False)
        except Empty:
            if len(self._noises) == 0:
                raise RuntimeError("No strain data after 10 seconds")
        else:
            # if we do, add it to our running list of strains
            self._strains.append((fname, strain))

            # create a blank array to fill out our noise
            # and idx arrays as we collect more data
            zeros = np.zeros_like(strain)
            self._noises = np.append(self._noises, zeros)
            self._covered_idx = np.append(self._covered_idx, zeros)

        return noise_prediction

    def process(self, package):
        # grab the noise prediction from the package
        # slice out the batch and channel dimensions,
        # which will both just be 1 for this pipeline
        package = package["output_0"]
        x = package.x.reshape(-1)
        if len(x) != self.step_size:
            raise ValueError(
                "Noise prediction is of wrong length {}".format(len(x))
            )

        # use the package request id to figure out where
        # in the blank noise array we need to insert
        # this prediction. Subtract the running index
        # of the total number of samples we've processed so far
        start = package.request_id * self.step_size - self._frame_idx
        self._noises[start : start + len(x)] = x

        # update our mask to indicate which parts of our
        # noise array have had inference performed on them
        self._covered_idx[start : start + len(x)] = 1

        # if we've completely performed inference on
        # an entire frame's worth of data, postprocess
        # the predictions and produce the cleaned estimate
        if self._covered_idx[: len(self._strains[0][1])].all():
            # pop out the earliest strain and filename and
            fname, strain = self._strains.pop(0)
            fname = os.path.basename(fname)
            timestamp, _ = _parse_frame_name(fname)

            # remove the data from both the running
            # noise array and the mask
            noise, self._noises = np.split(self._noises, [len(strain)])
            self._covered_idx = self._covered_idx[len(strain) :]
            self._frame_idx += len(noise)

            # now postprocess the noise and strain channels
            # TODO: do some sort of windowing before filtering?
            strain = self.preprocessor.filter(strain)
            # noise = self.preprocessor.uncenter(noise)
            noise = self.preprocessor.filter(noise)

            # remove the noise from the strain channel and
            # use it to create a timeseries we can write to .gwf
            cleaned = strain - noise
            timeseries = TimeSeries(
                cleaned,
                t0=timestamp,
                sample_rate=self.sample_rate,
                channel=self.channel_name,
            )

            # write the file and pass the written filename
            # to downstream processess
            write_fname = os.path.join(self.write_dir, fname)
            timeseries.write(write_fname)
            super().process(write_fname)
