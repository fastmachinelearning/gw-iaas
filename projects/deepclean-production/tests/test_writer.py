import logging
import os
import shutil
from queue import Empty, Queue
from unittest.mock import Mock

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from deepcleaner.utils import FrameWriter


@pytest.fixture(scope="session")
def write_dir():
    os.makedirs("tmp")
    yield "tmp"
    shutil.rmtree("tmp")


@pytest.fixture(scope="session")
def timestamp():
    return int("1" * 10)


@pytest.fixture(scope="session")
def fnames(write_dir, timestamp):
    witness_fnames, strain_fnames = [], []
    for i in range(10):
        fname = os.path.join(write_dir, "{}-{}-1.gwf")

        witness_fname = fname.format("witness", timestamp + i)
        with open(witness_fname, "w") as f:
            f.write("")

        witness_fnames.append(witness_fname)
        strain_fnames.append(fname.format("strain", timestamp + i))
    return witness_fnames, strain_fnames


@pytest.fixture(params=[8])  # , 64, 256])
def step_size(request):
    return request.param


@pytest.fixture(params=[2])  # , 6, 10])
def memory(request):
    return request.param


@pytest.fixture(params=[40 / 4096])  # , 0.5, 1])
def filter_pad(request):
    return request.param


def test_frame_writer(write_dir, fnames, step_size, memory, filter_pad):
    strain_q = Queue()
    preprocessor = Mock()
    preprocessor.uncenter = lambda x: x
    preprocessor.filter = lambda x: x

    writer = FrameWriter(
        write_dir,
        channel_name="dummy",
        step_size=step_size,
        sample_rate=4096.0,
        memory=memory,
        filter_pad=filter_pad,
        strain_q=strain_q,
        preprocessor=preprocessor,
        name="writer",
    )
    writer.logger = logging.getLogger()

    witness_fnames, strain_fnames = fnames
    num_steps = 4096 * len(witness_fnames) // step_size
    x = np.repeat(np.arange(num_steps), step_size)

    noises = np.split(x, num_steps)
    strains = np.split(x, len(strain_fnames))

    writer.strain_q.put(((witness_fnames[0], strain_fnames[0]), strains[0]))

    pad_samples = int(4096 * filter_pad)
    for i, x in enumerate(noises):
        writer.in_q.put((x,))

        noise = writer.get_package()
        package = Mock()
        package.x = noise[0]
        package.request_id = i
        writer.process({"aggregator": package})

        frame_end = (i + 1) * step_size
        if (frame_end / 2048) % 2 == 1:
            idx = (frame_end + 2048) // 4096
            if idx < len(strains):
                writer.strain_q.put(
                    ((witness_fnames[idx], strain_fnames[idx]), strains[idx])
                )

        if frame_end > pad_samples and (frame_end - pad_samples) % 4096 == 0:
            try:
                written, _ = writer.out_q.get(0.1)
            except Empty:
                raise ValueError(str(frame_end))

            idx = (frame_end - pad_samples) // 4096 - 1
            assert written == strain_fnames[idx]

            ts = TimeSeries.read(written, "dummy")
            assert (ts.value == 0).all()

            num_remaining_frames = min(memory, idx + 1)
            assert writer._covered_idx.sum() == (
                num_remaining_frames * 4096 + pad_samples
            )
            assert len(writer._noises) <= ((memory + 1) * 4096)
