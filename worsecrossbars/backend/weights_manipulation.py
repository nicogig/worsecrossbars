"""weights_manipulation:
A backend module used to map ANN weights to real-world resistance levels.
"""
import copy

import numpy as np
import tensorflow as tf


def bucketize_weights_layer(
    weights: tf.Tensor,
    hrs_lrs_ratio: float,
    number_conductance_levels: int,
    excluded_weights_proportion: float,
) -> tf.Tensor:
    """This function maps ANN weights to real-world, discrete resistance levels.

    Args:
      weights: A tensor of weights.
      hrs_lrs_ratio: The ratio between high and low resistance levels.
      number_conductance_levels: The number of conductance levels.
      excluded_weights_proportion: The proportion of weights to be excluded.

    Returns:
      discretised_w: A tensor of weights mapped to real-world, discrete resistance levels.
    """

    flattened_weights = tf.reshape(weights, [-1])

    # Casting to float64 because Tensorflow Metal's sort() function does not work properly with
    # float32 tensors
    flattened_weights = tf.cast(flattened_weights, tf.float64)

    sorted_weights = -tf.sort(-flattened_weights)

    # Casting back to float32
    sorted_weights = tf.cast(sorted_weights, tf.float32)

    # Finding minimum and maximum weight
    max_index = int(excluded_weights_proportion * tf.shape(sorted_weights)[0].numpy())
    w_max = sorted_weights[max_index]
    w_min = w_max / hrs_lrs_ratio

    cond_levels = np.linspace(
        w_min.numpy(), w_max.numpy(), number_conductance_levels, dtype=float
    ).tolist()

    indices = tf.raw_ops.Bucketize(input=weights, boundaries=cond_levels)

    discretised_w = copy.deepcopy(weights)
    mask = indices > len(cond_levels) - 1
    indices = tf.where(mask, len(cond_levels) - 1, indices)

    for index, cond_level in enumerate(cond_levels):
        discretised_w = tf.where(tf.equal(indices, index), cond_level, discretised_w)

    return discretised_w
