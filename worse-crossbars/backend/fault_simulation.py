import numpy as np
import copy
import gc
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from backend.weight_mapping import choose_extremes, create_weight_interval, discretise_weights

def weight_alterations(network_weights, fault_type=1, failure_percentage=0.2, extremes_list=[]):
    """
    weight_alterations:
        Alter the weights in a Neural Network to simulate a fault in a RRAM Crossbar Array.
    Inputs:
        -   network_weights: The weights as outputted by the training functions.
        -   fault_value: The value the devices are stuck at. By default, devices that cannot electroform are simulated, thus this value is 0.
        -   failure_percentage: The fail rate, expressed as a decimal value. Default: 0.2
    Output:
        -   The altered weights of the Neural Network.

    """

    # For the time being, only synapse parameters (i.e. weights, not neuron biases) are being altered. This is achieved by only modifying even-numbered layers,
    # given that, in a densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.
    
    altered_weights = copy.deepcopy(network_weights)
    
    for count, layer in enumerate(altered_weights):
        if count % 2 == 0:
            if fault_type == 1:
                fault_value = 0
            else:
                fault_value = extremes_list[count][fault_type - 2]
            indices = np.random.choice(layer.shape[1]*layer.shape[0], replace=False, 
                               size=int(layer.shape[1]*layer.shape[0]*failure_percentage))
            
            # Creating a sign mask to ensure that devices stuck at HRS/LRS retain the correct sign (i.e. that the associated weights remain negative if they were negative)
            signs_mask = np.sign(layer)

            layer[np.unravel_index(indices, layer.shape)] = fault_value * signs_mask[np.unravel_index(indices, layer.shape)]

    return altered_weights



def run_simulation(percentages_array, weights, number_of_simulations, network_model, dataset, fault_type=1, HRS_LRS_ratio=None, number_of_conductance_levels=10, excluded_weights_proportion=None):
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

    if fault_type not in (1, 2, 3):
        raise ValueError("fault_type is an illegal integer.")
    else:
        extremes_list = choose_extremes(weights, HRS_LRS_ratio, excluded_weights_proportion)
        weight_intervals = create_weight_interval(extremes_list, number_of_conductance_levels)
        weights = discretise_weights(weights, weight_intervals)

    accuracies = np.zeros(len(percentages_array))

    for simulation in range(number_of_simulations):

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
