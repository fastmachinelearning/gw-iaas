import numpy as np
import pytest

try:
    from hermes.quiver.streaming.streaming_output import Aggregator

    _has_tf = True
except ImportError as e:
    if "tensorflow" not in e:
        raise
    _has_tf = False


@pytest.mark.tensorflow
def test_aggregator(num_updates=10, update_size=8):
    if not _has_tf:
        raise ImportError("Can't test aggregator without TensorFlow installed")

    agg = Aggregator(num_updates=num_updates, update_size=update_size)

    # make sure that our shape checks raise the appropriate errors
    with pytest.raises(ValueError):
        # must have batch size of 1
        x = np.ones((2, num_updates * update_size))
        agg(x, 1)

    with pytest.raises(ValueError):
        # must have the appropriate update size
        x = np.ones((1, update_size * (num_updates - 1)))
        agg(x, 1)

    # run an input through the layer so that
    # it gets built, indicating that this is
    # the start of a new sequence
    x = np.ones((1, num_updates * update_size))
    y = agg(x, 1)

    # make sure that the update index is set to 1
    # and that the output matches the input, since
    # there has been nothing to aggregate it with
    assert agg.update_idx.numpy() == 1
    assert y.shape == (1, update_size)
    assert (y.numpy() == 1).all()

    # now run another input through on this sequence,
    # with the value incremented by one,
    # and make sure that the update index increments
    # and that the output is the average of the
    # first and second inputs
    y = agg(x + 1, 0)
    assert agg.update_idx.numpy() == 2
    assert (y.numpy() == 1.5).all()

    # verify once more that we have the average of all 3
    y = agg(x + 2, 0)
    assert (y.numpy() == 2).all()

    # now restart the sequence and make sure that
    # everything resets properly
    y = agg(x + 1, 1)
    assert agg.update_idx.numpy() == 1
    assert (y.numpy() == 2).all()
