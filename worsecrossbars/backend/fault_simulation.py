"""
fault_simulation:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""

import copy
import gc
import numpy as np
from worsecrossbars.backend.weight_mapping import choose_extremes
from worsecrossbars.backend.weight_mapping import create_weight_interval
from worsecrossbars.backend.weight_mapping import discretise_weights


def weight_alterations(network_weights, fault_type, failure_percentage, extremes_list):
    """
    This function takes in and modifies network weights to simulate the effect of faulty memristive
    devices being used in the physical implementation of the ANN.

    It should be noted that only synapse parameters (i.e. weights, not neuron biases) are being
    altered. This is achieved by only modifying even-numbered layers, given that, in a
    densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.

    Args:
      network_weights: Array containing the weights as outputted by the training functions.
      fault_type: Integer comprised between 1 and 3, indicates what fault is being simulated. A
        value of 1 indicates that the memristive devices are unable to electroform, a value of 2
        indicates that the devices are stuck in a high resistance state, and a value of 3 indicates
        that the memristors are stuck in a low resistance state.
      failure_percentage: Positive integer/float, percentage of devices affected by the fault
      extremes_list: List containing minimum and maximum weight values in a given layer

    altered_weights:
      Array containing the ANN weights altered to simulate the effect of the desired percentage of
      devices being affected by the specified fault.
    """

    if fault_type not in [1, 2, 3]:
        raise ValueError("\"fault_type\" argument should be an integer between 1 and 3.")

    if isinstance(failure_percentage, int):
        failure_percentage = float(failure_percentage)

    if not isinstance(failure_percentage, float) or failure_percentage < 0:
        raise ValueError("\"failure_percentage\" argument should be a positive real number.")

    altered_weights = copy.deepcopy(network_weights)

    for count, layer in enumerate(altered_weights):
        if count % 2 == 0:
            if fault_type == 1:
                fault_value = 0
            else:
                fault_value = extremes_list[count][fault_type - 2]
            indices = np.random.choice(layer.shape[1]*layer.shape[0], replace=False,
                                       size=int(layer.shape[1]*layer.shape[0]*failure_percentage))

            # Creating a sign mask to ensure that devices stuck at HRS/LRS retain the correct sign
            # (i.e. that the associated weights remain negative if they were negative)
            signs_mask = np.sign(layer)

            layer[np.unravel_index(indices, layer.shape)] = fault_value * \
                signs_mask[np.unravel_index(indices, layer.shape)]

    return altered_weights


def run_simulation(percentages_array, weights, number_of_simulations, network_model, dataset, fault_type=1, HRS_LRS_ratio=None, number_of_conductance_levels=None, excluded_weights_proportion=None):
    """
    run_simulation:
        Simulates a fault in a RRAM network with the given topology and weights, for a number of times.
    Inputs:
        -   percentages_array: A numpy array formed of decimal values representing the percentage of synapses in the network that are faulty.
        -   weights: The weights of the neural network.
        -   number_of_simulations: An integer representing the number of times the simulation will be run.
        -   network_model: A Keras model of the network.
        -   fault_type: The type of fault, expressed by an integer.
    Output:
        -   A list of average accuracies obtained by running the fault simulations "number_of_simulations" times.
    """

    if fault_type not in [1, 2, 3]:
        raise ValueError("\"fault_type\" argument should be an integer between 1 and 3.")

    extremes_list = choose_extremes(weights, HRS_LRS_ratio, excluded_weights_proportion)
    weight_intervals = create_weight_interval(extremes_list, number_of_conductance_levels)
    weights = discretise_weights(weights, weight_intervals)

    accuracies = np.zeros(len(percentages_array))

    for _ in range(number_of_simulations):

        accuracies_list = []

        for percentage in percentages_array:
            altered_weights = weight_alterations(weights, fault_type, percentage, extremes_list)

            # The "set_weights" function sets the ANN's weights to the values specified in the list of arrays "altered_weights"
            network_model.set_weights(altered_weights)
            accuracies_list.append(network_model.evaluate(dataset[1][0], dataset[1][1], verbose=0)[1])

        accuracies += np.array(accuracies_list)
        gc.collect()

    accuracies /= number_of_simulations

    return accuracies
