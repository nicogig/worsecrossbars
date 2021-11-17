"""
weight_mapping:
A backend module used to map ANN weights into real-world resistance levels.
"""

import copy
import numpy as np


def choose_extremes(network_weights, hrs_lrs_ratio, excluded_weights_proportion):
    """
    This function chooses the minimum and maximum discrete weights of the memristor ANN.

    Args:
      network_weights: The weights as outputted by the training functions.
      hrs_lrs_ratio: The desired HRS/LRS ratio.
      excluded_weights_proportion: The proportion of weights at the tails to be excluded.

    extremes_list:
      The list of extremes at each layer of the network.
    """

    extremes_list = []

    for count, layer_weights in enumerate(network_weights):

        if count % 2 == 0:
            array_weights = layer_weights.flatten()
            w_abs = np.abs(array_weights)
            w_abs[::-1].sort()
            size = w_abs.size
            index = int(excluded_weights_proportion * size)
            w_max = w_abs[index]
            w_min = w_max / hrs_lrs_ratio
            extremes_list.append((w_max, w_min))
        else:
            extremes_list.append((None,))

    return extremes_list


def create_weight_interval(list_of_extremes, number_of_levels):
    """
    This function creates an evenly spaced weight interval.

    Args:
      list_of_extremes: The list of extremes in each layer.
      number_of_levels: The number of weights needed.

    weight_interval_list:
      A list of linearly spaced weight intervals for the synaptic layers.
    """

    weight_interval_list = []

    for count, element in enumerate(list_of_extremes):
        if count % 2 == 0:
            weight_interval_list.append(
                np.linspace(element[1], element[0], number_of_levels)
            )

    return weight_interval_list


def discretise_weights(network_weights, network_weight_intervals):
    """
    This function alters the weights in the network so that they conform to the list of allowed
    weights.

    Args:
      network_weights: The weights as output by the training functions.
      network_weight_intervals: A list of spaced weight intervals.

    discretised_weights:
      The altered network weights, now discretised.
    """

    discretised_weights = copy.deepcopy(network_weights)
    weight_int_count = -1

    for count, layer_weights in enumerate(discretised_weights):

        if count % 2 == 0:
            weight_int_count += 1
            original_shape = layer_weights.shape
            layer_weights = layer_weights.flatten()
            req_int = network_weight_intervals[weight_int_count]
            req_int = np.concatenate((np.negative(req_int)[::-1], req_int), axis=None)
            index = np.searchsorted(req_int, layer_weights)
            mask = index > len(req_int) - 1
            index[mask] = len(req_int) - 1
            index_new = np.array(
                [
                    index[i] - 1
                    if abs(req_int[index[i] - 1] - layer_weights[i])
                    < abs(req_int[index[i]] - layer_weights[i])
                    else index[i]
                    for i in range(len(index))
                ]
            )
            layer_weights = np.array([req_int[_] for _ in index_new])
            layer_weights = np.reshape(layer_weights, original_shape)
            discretised_weights[count] = layer_weights

    return discretised_weights
