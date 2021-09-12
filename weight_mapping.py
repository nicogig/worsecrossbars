import numpy as np

def choose_extremes(W, HRS_LRS_ratio, excluded_weights_proportion):
    """
    choose_extremes:
        Choose the minimum and maximum discrete weights
    Inputs:
        -   W, an array containing all the continuous weights
            in a synaptic layer.
        -   HRS_LRS_ratio: the ratio desired.
        -   excluded_weights_proportion:  proportion of excluded 
            synaptic weights with the largest absolute values.
    Output:
        - A tuple, (w_max, w_min), with the maximum and minimum 
        discrete weights.
    """
    
    W_abs = np.absolute(W)
    W_abs_sort = np.argsort(-W_abs)
    s = W_abs_sort.size
    index = int(excluded_weights_proportion * s)
    w_max = W_abs_sort(index)
    w_min = w_max / HRS_LRS_ratio
    return (w_max, w_min)

