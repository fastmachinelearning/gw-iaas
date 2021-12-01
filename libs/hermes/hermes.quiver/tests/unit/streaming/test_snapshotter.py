from collections import OrderedDict

import numpy as np
import pytest

try:
    from hermes.quiver.streaming.streaming_input import Snapshotter

    _has_tf = True
except ImportError as e:
    if "tensorflow" not in e:
        raise
    _has_tf = False


@pytest.mark.tensorflow
def test_snapshotter(snapshot_size=100, update_size=10):
    if not _has_tf:
        raise ImportError("Can't test aggregator without TensorFlow installed")

    snapshotter = Snapshotter(
        snapshot_size, OrderedDict([("a", 6), ("b", 4), ("c", 8)])
    )

    # make sure our shape checks catch any inconsistencies
    with pytest.raises(ValueError):
        # channel dimension must equal the total number of input channels
        x = np.ones((1, 17, update_size))
        snapshotter(x, 1)

    with pytest.raises(ValueError):
        # can't support batching
        x = np.ones((2, 18, update_size))
        snapshotter(x, 1)

    # now run an input through as a new sequence and
    # make sure we get the appropriate number of outputs
    x = np.ones((1, 18, update_size))
    y = snapshotter(x, 1)
    assert len(y) == 3

    # now make sure that each snapshot has the appropriate shape
    # and has 0s everywhere except for the most recent update
    for y_, channels in zip(y, [6, 4, 8]):
        y_ = y_.numpy()
        assert y_.shape == (1, channels, snapshot_size)
        assert (y_[:, :, :-update_size] == 0).all()
        assert (y_[:, :, -update_size:] == 1).all()

    # make another update and verify that the snapshots
    # all contain the expected update values
    y = snapshotter(x + 1, 0)
    for y_ in y:
        y_ = y_.numpy()
        assert (y_[:, :, : -update_size * 2] == 0).all()
        for i in range(2):
            start = -update_size * (2 - i)
            stop = (-update_size * (1 - i)) or None
            assert (y_[:, :, start:stop] == i + 1).all()

    # reset the sequence and make sure that the
    # snapshot resets and updates properly
    y = snapshotter(x + 2, 1)
    for y_ in y:
        y_ = y_.numpy()
        assert (y_[:, :, :-update_size] == 0).all()
        assert (y_[:, :, -update_size:] == 3).all()
