import numpy as np
import copy
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

def weight_alterations(network_weights, fault_value=0, failure_percentage=0.2):
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
            indices = np.random.choice(layer.shape[1]*layer.shape[0], replace=False, 
                               size=int(layer.shape[1]*layer.shape[0]*failure_percentage))
            layer[np.unravel_index(indices, layer.shape)] = fault_value
    
    return altered_weights



def run_simulation(percentages_array, weights, number_of_simulations):
    """
    
    """

    accuracies = np.zeros(len(percentages))

    MNIST_MLP = generator_functions[number_of_hidden_layers]()
    MNIST_MLP.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    for count, weights in enumerate(weights_list):

        accuracies_list = []

        for percentage in percentages:
            
            altered_weights = alteration_functions[fault_type](weights, percentage)
            
            # The "set_weights" function sets the ANN's weights to the values specified in the list of arrays "altered_weights"
            MNIST_MLP.set_weights(altered_weights)
            accuracies_list.append(MNIST_MLP.evaluate(MNIST_dataset[1][0], MNIST_dataset[1][1], verbose=0)[1])

        accuracies += np.array(accuracies_list)

        if (count+1) % 5 == 0 and count != 0:
            print("Finished evaluating weight set #{}.".format(count+1))
        
        gc.collect()

    return accuracies /= len(weights_list)
