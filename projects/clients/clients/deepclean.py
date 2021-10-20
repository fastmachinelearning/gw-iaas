import logging
import os
from multiprocessing import Queue
from queue import Empty
from typing import Optional, Sequence

import numpy as np
from gwpy.timeseries import TimeSeries

from hermes.gwftools import FrameCrawler, FrameLoader, GCSFrameDownloader
from hermes.gwftools.gwftools import _parse_frame_name
from hermes.stillwater import InferenceClient, PipelineProcess
from hermes.typeo import typeo


class Preprocessor:
    def __init__(self, preproc_params):
        pass


class FrameCollector(PipelineProcess):
    def __init__(
        self,
        write_dir: str,
        channel_name: str,
        step_size: int,
        sample_rate: float,
        strain_q: Queue,
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
        x = package.x.reshape(-1)
        if len(x) != self.step_size:
            raise ValueError(
                "Noise prediction is of wrong length {}".format(
                    len(x)
                )
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

            # now postprocess the noise channel
            noise = self.preprocessor.uncenter(noise)
            noise = self.preprocessor.bandpass(noise)

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


@typeo("DeepClean client")
def main(
    data_dir: str,
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    inference_rate: float,
    channels: Sequence[str],
    sequence_id: int,
    url: str,
    model_name: str,
    model_version: int = 1,
    t0: Optional[float] = None,
    length: Optional[float] = None,
    preprocess_pkl: Optional[str] = None,
    timeout: Optional[float] = None,
) -> None:
    # source for frame filenames will be different
    # depending on whether our data is local or in
    # the cloud
    if data_dir.startswith("gs://"):
        # for cloud data, download to local files and
        # pass the names of these downloaded files to
        # frame loading process
        root = data_dir.replace("gs://", "")
        fname_source = GCSFrameDownloader(
            root=root, t0=t0, length=length, name="fname-source"
        )

        # we don't need these frames locally once they're
        # loaded, so leave that to the frame loader
        remove = True
    else:
        # for local replay data, create a process which
        # monitors a local directory for new frames and
        # passes the names of those frames to the frame loader
        fname_source = FrameCrawler(
            data_dir, timeout=timeout, name="fname-source"
        )

        # these are likely being managed by a process
        # that isn't ours to interfere with, so don't
        # delete the files onces we're done with them
        remove = False

    # if we specified a DeepClean preprocessing
    # pickle, build a callable object which can
    # perform the requisite preprocessing
    if preprocess_pkl is not None:
        preprocessor = Preprocessor(preprocess_pkl)
    else:
        preprocessor = None

    # we want to be able to pass our strain data
    # directly to the postprocessing process, so
    # create a queue which our frame loader can
    # use to pass strain data and filenames
    strain_q = Queue()
    data_loader = FrameLoader(
        chunk_size=int(kernel_length * sample_rate),
        step_size=int(stride_length * sample_rate),
        sample_rate=sample_rate,
        channels=channels,
        t0=t0,
        length=length,
        sequence_id=sequence_id,
        preprocessor=preprocessor,
        remove=remove,
        strain_q=strain_q,
        rate=inference_rate,
        name="frame-loader",
    )

    # now create a client process which will take
    # the streams output by the data loader
    # and package them up for the server
    # TODO: include an ignore_streams arg
    # to the inference client in case we want
    # to make requests to the streaming ensemble
    # in an arbitrary order
    client = InferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        profile=False,
        name="client",
    )

    # build a pipeline connecting all the processes
    with fname_source >> data_loader >> client as pipeline:
        for fname in pipeline:
            logging.info(f"Processed frame {fname}")


if __name__ == "__main__":
    main()
