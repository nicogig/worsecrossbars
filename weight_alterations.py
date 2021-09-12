import numpy as np
import copy

def cannot_electroform(network_weights, failure_percentage=0.2):
    """

    """

    # For the time being, only synapse parameters (i.e. weights, not neuron biases) are being altered. This is achieved by only modifying even-numbered layers,
    # given that, in a densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.
    
    altered_weights = copy.deepcopy(network_weights)
    
    for count, layer in enumerate(altered_weights):
        if count % 2 == 0:
            indices = np.random.choice(layer.shape[1]*layer.shape[0], replace=False, 
                               size=int(layer.shape[1]*layer.shape[0]*failure_percentage))
            layer[np.unravel_index(indices, layer.shape)] = 0
    
    return altered_weights



def stuck_at_HRS(network_weights, failure_percentage=0.2):
    """
    
    """

    # For the time being, only synapse parameters (i.e. weights, not neuron biases) are being altered. This is achieved by only modifying even-numbered layers,
    # given that, in a densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.

    pass



def stuck_at_LRS(network_weights, failure_percentage=0.2):
    """
    
    """

    # For the time being, only synapse parameters (i.e. weights, not neuron biases) are being altered. This is achieved by only modifying even-numbered layers,
    # given that, in a densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.

    pass