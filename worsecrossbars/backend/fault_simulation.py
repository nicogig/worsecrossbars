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
      fault_type: String, indicates which fault is being simulated.
      failure_percentage: Positive integer/float, percentage of devices affected by the fault.
      extremes_list: List containing minimum and maximum weight values in a given layer.

    altered_weights:
      Array containing the ANN weights altered to simulate the effect of the desired percentage of
      devices being affected by the specified fault.
    """

    if isinstance(failure_percentage, int):
        failure_percentage = float(failure_percentage)

    if not isinstance(failure_percentage, float) or failure_percentage < 0:
        raise ValueError("\"failure_percentage\" argument should be a positive real number.")

    altered_weights = copy.deepcopy(network_weights)

    for count, layer in enumerate(altered_weights):

        if count % 2 == 0:
            if fault_type == "STUCK_ZERO":
                fault_value = 0
            elif fault_type == "STUCK_HRS":
                fault_value = extremes_list[count][0]
            else:
                fault_value = extremes_list[count][1]

            indices = np.random.choice(layer.shape[1]*layer.shape[0], replace=False,
                                       size=int(layer.shape[1]*layer.shape[0]*failure_percentage))

            # Creating a sign mask to ensure that devices stuck at HRS/LRS retain the correct sign
            # (i.e. that the associated weights remain negative if they were negative)
            signs_mask = np.sign(layer)

            layer[np.unravel_index(indices, layer.shape)] = fault_value * \
                signs_mask[np.unravel_index(indices, layer.shape)]

    return altered_weights


def run_simulation(percentages_array, weights, network_model, dataset, simulation_parameters):
    """
    """

    extremes_list = choose_extremes(weights, simulation_parameters["HRS_LRS_ratio"],
                                    simulation_parameters["excluded_weights_proportion"])
    weight_intervals = create_weight_interval(extremes_list,
                                              simulation_parameters["number_of_conductance_levels"])
    weights = discretise_weights(weights, weight_intervals)

    accuracies = np.zeros(len(percentages_array))

    for _ in range(simulation_parameters["number_of_simulations"]):

        accuracies_list = []

        for percentage in percentages_array:
            altered_weights = weight_alterations(weights, simulation_parameters["fault_type"],
                                                 percentage, extremes_list)

            # The "set_weights" function sets the ANN's weights to the values specified in the
            # list of arrays "altered_weights"
            network_model.set_weights(altered_weights)
            accuracies_list.append(network_model.evaluate(dataset[1][0],
                                   dataset[1][1], verbose=0)[1])

        accuracies += np.array(accuracies_list)
        gc.collect()

    accuracies /= simulation_parameters["number_of_simulations"]

    return accuracies
