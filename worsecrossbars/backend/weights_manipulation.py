"""weights_manipulation:
A backend module used to map ANN weights to real-world resistance levels.
"""
import copy

import tensorflow as tf

import numpy as np

import json


def bucketize_weights_layer(
    w: tf.Tensor,
    hrs_lrs_ratio: float,
    number_conductance_levels: int,
    excluded_weights_proportion: float,
) -> tf.Tensor:
    """"""

    with open("w.json", "w") as fout:
        json.dump(w.numpy().tolist(), fout)

    flattened_weights = tf.reshape(w, [-1])

    with open("flattened.json", "w") as fout:
        json.dump(flattened_weights.numpy().tolist(), fout)

    sorted_weights = tf.sort(flattened_weights)

    with open("sorted.json", "w") as fout:
        json.dump(sorted_weights.numpy().tolist(), fout)

    # sorted_weights = -tf.sort(-tf.reshape(w, [-1]))
    # sorted_weights = tf.sort(-tf.reshape(-w, [-1]))
    max_index = int(excluded_weights_proportion * tf.shape(sorted_weights)[0].numpy())
    w_max = sorted_weights[max_index]
    w_min = w_max / hrs_lrs_ratio

    print(sorted_weights)
    # print(max_index)

    print(w_max)
    print(w_min)

    cond_levels = np.linspace(
        w_min.numpy(), w_max.numpy(), number_conductance_levels, dtype=float
    ).tolist()
    print(cond_levels)
    indices = tf.raw_ops.Bucketize(input=w, boundaries=cond_levels)

    # print(indices)

    discretised_w = copy.deepcopy(w)
    mask = indices > len(cond_levels) - 1
    indices = tf.where(mask, len(cond_levels) - 1, indices)

    for index, cond_level in enumerate(cond_levels):
        discretised_w = tf.where(tf.equal(indices, index), cond_level, discretised_w)

    return discretised_w
