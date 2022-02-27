"""weights_manipulation:
A backend module used to map ANN weights to real-world resistance levels.
"""
import copy

import tensorflow as tf

import numpy as np


def bucketize_weights_layer(
    w: tf.Tensor,
    hrs_lrs_ratio: float,
    number_conductance_levels: int,
    excluded_weights_proportion: float,
) -> tf.Tensor:
    """"""

    flattened_weights = tf.reshape(w, [-1])

    # Casting to float64 because Tensorflow Metal's sort() function does not work properly with
    # float32 tensors
    sorted_weights = tf.sort(tf.cast(flattened_weights, tf.float64))

    # Casting back to float32
    sorted_weights = tf.cast(sorted_weights, tf.float32)

    # Finding minimum and maximum weight
    max_index = int(excluded_weights_proportion * tf.shape(sorted_weights)[0].numpy())
    w_max = sorted_weights[max_index]
    w_min = w_max / hrs_lrs_ratio

    cond_levels = np.linspace(
        w_min.numpy(), w_max.numpy(), number_conductance_levels, dtype=float
    ).tolist()
    indices = tf.raw_ops.Bucketize(input=w, boundaries=cond_levels)

    discretised_w = copy.deepcopy(w)
    mask = indices > len(cond_levels) - 1
    indices = tf.where(mask, len(cond_levels) - 1, indices)

    for index, cond_level in enumerate(cond_levels):
        discretised_w = tf.where(tf.equal(indices, index), cond_level, discretised_w)

    return discretised_w
