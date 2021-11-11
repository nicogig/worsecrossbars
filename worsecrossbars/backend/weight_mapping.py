"""
weight_mapping:
A backend module used to map ANN weights into real-world resistance levels.
"""

import copy
import numpy as np


def choose_extremes(network_weights, hrs_lrs_ratio, excluded_weights_proportion):
    """
    choose_extremes:
        Choose the minimum and maximum discrete weights
    Inputs:
        -   network_weights: The weights as outputted by the training functions.
        -   HRS_LRS_ratio: the ratio desired.
        -   excluded_weights_proportion:  proportion of excluded
            synaptic weights with the largest absolute values.
    Output:
        - A list of tuples, (w_max, w_min), with the maximum and minimum
        discrete weights in a layer.
    """

    return_list = []
    for count, layer_weights in enumerate(network_weights):
        if count % 2 == 0:
            array_weights = layer_weights.flatten()
            w_abs = np.abs(array_weights)
            w_abs[::-1].sort()
            size = w_abs.size
            index = int(excluded_weights_proportion * size)
            w_max = w_abs[index]
            w_min = w_max / hrs_lrs_ratio
            return_list.append((w_max, w_min))
        else:
            return_list.append((None, ))
    return return_list


def create_weight_interval(list_of_extremes, number_of_levels):
    """
    create_weight_interval:
        Create an evenly spaced weight interval.
    Inputs:
        -   list_of_extremes: A list of tuples, as returned by choose_extremes()
        -   number_of_levels: The number of weights needed.
    Output:
        -   A list of lists of evenly spaced weights.
    """

    return_list = []
    for count, element in enumerate(list_of_extremes):
        if count % 2 == 0:
            return_list.append(np.linspace(element[1], element[0], number_of_levels))
    return return_list


def discretise_weights(network_weights, network_weight_intervals):
    """
    discretise_weights:
        Alter the weights in the network so that they conform to the list of allowed weights.
    Inputs:
        -   network_weights: The weights as outputted by the training functions.
        -   network_weight_intervals: A list of lists of evenly spaced weights.
        One list per synaptic layer.
    Output:
        -   The altered network weights, now discretised.
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
                [index[i] - 1 if abs(req_int[index[i] - 1] - layer_weights[i]) < \
                abs(req_int[index[i]] - layer_weights[i]) else index[i] for i in range(len(index))]
            )
            layer_weights = np.array([req_int[_] for _ in index_new])
            layer_weights = np.reshape(layer_weights, original_shape)
            discretised_weights[count] = layer_weights

    return discretised_weights
