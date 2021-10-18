import os
from multiprocessing import Queue
from unittest.mock import Mock

import numpy as np
import pytest
from client.deepclean import FrameCollector
from gwpy.timeseris import TimeSeries, TimeSeriesDict

from hermes.gwftools import gwftools as gwf


@pytest.fixture(scope="session")
def tstamp():
    return 1313883327


@pytest.fixture(scope="session")
def fformat():
    return "H-H1_llhoft-{}-1.gwf"


def soft_makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def soft_rmdir(d):
    if os.path.exists(d):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)


@pytest.fixture(scope="session")
def read_dir():
    read_dir = "tmp-read"
    soft_makedirs(read_dir)
    yield read_dir
    soft_rmdir(read_dir)


@pytest.fixture(scope="session")
def write_dir():
    write_dir = "tmp-write"
    soft_makedirs(write_dir)
    yield write_dir
    soft_rmdir(write_dir)


@pytest.fixture(scope="session")
def fnames(tstamp, fformat, read_dir):
    x = np.zeros((4096,))
    fnames = []
    for i in range(10):
        td = {}
        for channel in ["x", "y"]:
            td[channel] = TimeSeries(
                x + i, t0=tstamp + i, sample_rate=4096, channel=channel
            )
        td = TimeSeriesDict(td)

        fname = os.path.join(read_dir, fformat.format(tstamp + i))
        td.write(fname)
        fnames.append(fname)

    yield fnames


@pytest.mark.parametrize(
    "chunk_size,step_size,sample_rate",
    [
        (256, 256, 4096),
        (256, 256, 2048),
        (256, 32, 4096),
        (256, 31, 4096),
    ],
)
def test_frame_collector(
    chunk_size, step_size, sample_rate, fnames, read_dir, write_dir
):
    strain_q = Queue()
    loader = gwf.FrameLoader(
        chunk_size=chunk_size,
        step_size=step_size,
        sample_rate=sample_rate,
        channels=["x", "y"],
        remove=True,
        strain_q=strain_q,
        name="loader",
    )

    collector = FrameCollector(
        write_dir,
        channel_name="y",
        step_size=step_size,
        sample_rate=sample_rate,
        strain_q=strain_q,
        name="collector",
        preprocessor=Mock(uncenter=lambda x: x, bandpass=lambda x: x),
    )

    with loader >> collector as pipeline:
        for f in pipeline:
            ts = TimeSeries.read(f, channel="y")
            assert len(ts.value) == sample_rate
            assert (ts.value == 0).all()
            os.remove(f)
