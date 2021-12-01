import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(name="Aggregator")
class Aggregator(tf.keras.layers.Layer):
    def __init__(
        self, update_size: int, num_updates: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.update_size = update_size
        self.num_updates = num_updates

    def build(self, input_shape) -> None:
        snapshot_size = self.update_size * self.num_updates

        if input_shape[0] is None:
            raise ValueError("Must specify batch dimension")
        if input_shape[0] != 1:
            # TODO: support batching
            raise ValueError("Batching not currently supported")
        if input_shape[-1] != snapshot_size:
            raise ValueError(
                "Expected input update of {} samples, but "
                "found {}".format(snapshot_size, input_shape[-1])
            )

        self.update_idx = self.add_weight(
            name="update_idx", shape=[], dtype=tf.float32, initializer="zeros"
        )

        snapshot_shape = [input_shape[0], snapshot_size - self.update_size]
        if len(input_shape) == 3:
            snapshot_shape.insert(1, input_shape[1])
        elif len(input_shape) > 3:
            raise ValueError(
                "Unsupported number of input dimensions {}".format(
                    len(input_shape)
                )
            )

        self.snapshot = self.add_weight(
            name="snapshot",
            shape=snapshot_shape,
            dtype=tf.float32,
            initializer="zeros",
        )

        self.update = tf.zeros((1, self.update_size), dtype=tf.float32)
        self.normalizer = tf.constant(
            np.repeat(np.arange(self.num_updates), self.update_size)[::-1] + 1,
            dtype=tf.float32,
        )

    def call(self, x, sequence_start):
        snapshot = (1.0 - sequence_start) * self.snapshot
        update_idx = (1.0 - sequence_start) * self.update_idx + 1

        snapshot = tf.concat([snapshot, self.update], axis=-1)
        weights = tf.clip_by_value(self.normalizer, 0, update_idx)
        snapshot += (x - snapshot) / weights

        output, snapshot = tf.split(
            snapshot,
            [self.update_size, self.update_size * (self.num_updates - 1)],
            axis=-1,
        )

        self.snapshot.assign(snapshot)
        self.update_idx.assign(update_idx)
        return output
