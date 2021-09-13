import numpy as np
import copy

def choose_extremes(network_weights, HRS_LRS_ratio, excluded_weights_proportion):
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
            array_weights = np.flatten(layer_weights)
            W_abs = np.abs(array_weights)
            W_abs_sort = np.argsort(-W_abs)
            s = W_abs_sort.size
            index = int(excluded_weights_proportion * s)
            w_max = W_abs_sort(index)
            w_min = w_max / HRS_LRS_ratio
            return_list.append((w_max, w_min))
        else:
            return_list.append((,))
    return return_list



def create_weight_interval(list_of_extremes, no_of_weights):
    """
    create_weight_interval:
        Create an evenly spaced weight interval.
    Inputs:
        -   list_of_extremes: A list of tuples, as returned by choose_extremes()
        -   no_of_weights: The number of weights needed.
    Output:
        -   A list of lists of evenly spaced weights.
    """

    return_list = []
    for count, element in enumerate(list_of_extremes):
        if count % 2 == 0:
            return_list.append(np.linspace(element[1], element[0], no_of_weights))
    return return_list



def discretise_weights(network_weights, network_weight_intervals):
    """
    discretise_weights:
        Alter the weights in the network so that they conform to the list of allowed weights.
    Inputs:
        -   network_weights: The weights as outputted by the training functions.
        -   network_weight_intervals: A list of lists of evenly spaced weights. One list per synaptic layer.
    Output:
        -   The altered network weights, now discretised.
    """
    altered_weights = copy.deepcopy(network_weights)
    index = 0
    for count, layer_weights in enumerate(altered_weights):
        if count % 2 == 0:
            index += 1
            original_shape = layer_weights.shape
            layer_weights = np.flatten(layer_weights)
            weight_intervals_as_numpy = np.array(network_weight_intervals[index])
            layer_weights = weight_intervals_as_numpy[abs(layer_weights[None, :] - weight_intervals_as_numpy[:, None]).argmin(axis=0)]
            layer_weights = np.reshape(layer_weights, original_shape)
    
    return altered_weights
