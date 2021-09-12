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
    for list_weights in network_weights:
        array_weights = np.array(list_weights)
        W_abs = np.abs(array_weights)
        W_abs_sort = np.argsort(-W_abs)
        s = W_abs_sort.size
        index = int(excluded_weights_proportion * s)
        w_max = W_abs_sort(index)
        w_min = w_max / HRS_LRS_ratio
        return_list.append((w_max, w_min))
    return return_list

