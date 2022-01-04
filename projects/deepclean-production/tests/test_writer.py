import logging
import os
import shutil
from queue import Empty, Queue
from unittest.mock import Mock

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from deepcleaner.utils import FrameWriter


@pytest.fixture
def write_dir():
    yield "tmp"
    shutil.rmtree("tmp")


@pytest.fixture
def timestamp():
    return "1" * 10


@pytest.fixture
def fnames(write_dir, timestamp):
    witness_fnames, strain_fnames = [], []
    for i in range(10):
        fname = os.path.join(write_dir, "{}_{}-1.gwf")

        witness_fname = fname.format("witness", timestamp)
        with open(witness_fname, "w") as f:
            f.write("")

        witness_fnames.append(witness_fname)
        strain_fnames.append(fname.format("strain", timestamp))
    return witness_fnames, strain_fnames


@pytest.fixture(params=[8, 64, 256])
def step_size(request):
    return request.param


@pytest.fixture(params=[2, 6, 10])
def memory(request):
    return request.param


@pytest.fixture(params=[0.1, 0.5, 1])
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
    )
    writer.logger = logging.getLogger()

    witness_fnames, strain_fnames = fnames
    num_steps = 4096 * len(witness_fnames) // step_size
    x = np.repeat(np.arange(num_steps), step_size)

    noises = np.split(x, num_steps)
    strains = np.split(x, len(strain_fnames))

    for x in noises:
        package = Mock()
        package.x = x
        writer.in_q.put({"aggregator": package})

    for (w_fname, s_fname, x) in zip(witness_fnames, strain_fnames, strains):
        writer.strain_q.put(((w_fname, s_fname), x))

    for i in range(num_steps):
        noise = writer.get_package()
        if i < len(strain_fnames):
            assert len(writer._strains) == (i + 1)
        writer.process(noise)

    for fname in strain_fnames:
        try:
            written, _ = writer.out_q.get_nowait()
        except Empty:
            raise ValueError
        assert written == fname

        ts = TimeSeries.read(fname, "dummy")
        assert (ts.value == 0).all()

    with pytest.raises(Empty):
        writer.out_q.get_nowait()
