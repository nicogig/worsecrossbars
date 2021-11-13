"""
custom_layers:
A backend module which implements a custom-made Keras AWGN layer.
"""

import tensorflow as tf
import numpy as np


class AWGN(tf.keras.layers.Dense): # pylint: disable=too-few-public-methods
    """
    -
    """

    def __init__(self, units, activation, noise_variance, noise_mean=0, **kwargs):
        super().__init__(units, activation, **kwargs)
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance

    def call(self, inputs):
        """
        -
        """

        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        rank = inputs.shape.rank
        noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_variance), self.kernel.shape)

        if rank == 2 or rank is None:

            if isinstance(inputs, tf.SparseTensor):
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                ids = tf.SparseTensor(indices=inputs.indices, values=inputs.indices[:, 1],
                                      dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(self.kernel+noise, ids, weights,
                                                        combiner='sum')
            else:
                outputs = tf.raw_ops.MatMul(a=inputs, b=self.kernel+noise)

        else:

            outputs = tf.tensordot(inputs, self.kernel+noise, [[rank - 1], [0]])

            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
