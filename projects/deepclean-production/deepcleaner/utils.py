import logging
import os
import pickle
import sys
import time
from queue import Empty
from typing import TYPE_CHECKING, Optional

import numpy as np
from gwpy.timeseries import TimeSeries
from scipy import signal

from hermes.gwftools.gwftools import (
    FrameCrawler,
    FrameLoader,
    _parse_frame_name,
)
from hermes.stillwater import PipelineProcess

if TYPE_CHECKING:
    from multiprocessing import Queue


def get_logger(filename: Optional[str] = None, verbose: bool = False):
    kwargs = {
        "level": logging.DEBUG if verbose else logging.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    if filename is not None:
        kwargs["filename"] = filename
        kwargs["filemode"] = "w"
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

        self.mean = params["mean"][1:]
        self.std = params["std"][1:]

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
        memory: float,
        filter_pad: float,
        strain_q: "Queue",
        throw_away: Optional[int] = None,
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
        self.past_samples = int(memory * sample_rate)
        self.future_samples = int(filter_pad * sample_rate)
        self.throw_away = throw_away
        self.strain_q = strain_q
        self.preprocessor = preprocessor

        self._strains = []
        self._noises = np.array([])
        self._covered_idx = np.array([])
        self._frame_idx = 0
        self._thrown_away = 0

    def get_package(self):
        # now get the next inferred noise estimate
        noise_prediction = super().get_package()

        # first see if we have any new strain data to collect
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
            self.logger.debug(f"Adding strain file {fname} to strains")
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
        package = package["aggregator"]
        self.logger.debug(
            "Received response for package {}".format(package.request_id)
        )
        x = package.x.reshape(-1)
        if len(x) != self.step_size:
            raise ValueError(
                "Noise prediction is of wrong length {}".format(len(x))
            )

        if self.throw_away is not None and self._thrown_away < self.throw_away:
            self._thrown_away += 1
            self.logger.debug(
                "Throwing away response for package {}".format(
                    package.request_id
                )
            )
            if self._thrown_away == self.throw_away:
                self.logger.debug("Done with throwaway responses")
            return

        # use the package request id to figure out where
        # in the blank noise array we need to insert
        # this prediction. Subtract the running index
        # of the total number of samples we've processed so far
        offset = max(self._frame_idx - self.past_samples, 0)
        start = int(package.request_id * self.step_size - offset)
        if self.throw_away is not None:
            start -= int(self.throw_away * self.step_size)
        self._noises[start : start + len(x)] = x

        # update our mask to indicate which parts of our
        # noise array have had inference performed on them
        self._covered_idx[start : start + len(x)] = 1
        if len(self._strains) == 0:
            self.logger.debug("No strains to clean, continuing")
            return

        # if we've completely performed inference on
        # an entire frame's worth of data, postprocess
        # the predictions and produce the cleaned estimate
        past_samples = min(self.past_samples, self._frame_idx)
        limit = (
            past_samples + len(self._strains[0][1]) + self.future_samples
        )
        if self._covered_idx[:limit].all():
            # pop out the earliest strain and filename and
            (witness_fname, strain_fname), strain = self._strains.pop(0)
            self.logger.debug("Cleaning strain file " + strain_fname)
            fname = os.path.basename(strain_fname)
            timestamp, _ = _parse_frame_name(fname)

            # remove the data from both the running
            # noise array and the mask

            # now postprocess the noise and strain channels
            noise = self.preprocessor.uncenter(self._noises)
            noise = self.preprocessor.filter(noise)
            noise = noise[past_samples:past_samples + len(strain)]

            self._frame_idx += len(strain)
            if (self._covered_idx.sum() - self.future_samples) > self.past_samples:
                self._noises = self._noises[len(strain):]
                self._covered_idx = self._covered_idx[len(strain):]

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
            latency = time.time() - os.stat(witness_fname).st_mtime
            super().process((write_fname, latency))


class TwoFileFrameCrawler(FrameCrawler):
    def __init__(
        self,
        witness_data_dir: str,
        strain_data_dir: str,
        timeout: float,
        N: Optional[int] = None,
        start_first: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            witness_data_dir, timeout, N=N, start_first=start_first, **kwargs
        )
        self.strain_data_dir = strain_data_dir

    def _get_fname(self):
        fnames = []
        for data_dir in [self.data_dir, self.strain_data_dir]:
            if data_dir == self.data_dir:
                pattern = self.pattern
            else:
                pattern = self.pattern.replace("Detchar", "HOFT")

            fname = os.path.join(data_dir, pattern.format(self.timestamp))
            self._wait_until_exists(fname)
            fnames.append(fname)
        return tuple(fnames)


class TwoFileFrameLoader(FrameLoader):
    def get_next_frame(self) -> np.ndarray:
        fnames = super(FrameLoader, self).get_package()
        witness_fname, strain_fname = fnames

        witness_data = self.load_frame_file(witness_fname, self.channels[1:])
        strain_data = self.load_frame_file(strain_fname, self.channels[:1])
        self.strain_q.put((fnames, strain_data[0]))

        if self.preprocessor is not None:
            witness_data = self.preprocessor(witness_data)
        if self._idx == 0:
            time.sleep(0.01)

        self.logger.debug(
            "Loaded frame files {} and {}".format(witness_fname, strain_fname)
        )
        return witness_data.astype("float32")

    def process(self, package):
        self.logger.debug(
            "Sending package with request id {}".format(package.request_id)
        )
        super().process(package)
