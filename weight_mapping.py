import numpy as np

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
    for element in list_of_extremes:
        return_list.append(np.linspace(element[1], element[0], no_of_weights))
    return return_list

