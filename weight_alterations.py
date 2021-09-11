import numpy as np

def cannot_electroform(network_weights, failure_percentage=0.2):

    # For the time being, only synapse parameters (i.e. weights, not neuron biases) are being altered. This is achieved by only modifying even-numbered layers,
    # given that, in a densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.
    
    for count, layer in enumerate(network_weights):
        if count % 2 == 0:
            indices = np.random.choice(layer.shape[1]*layer.shape[0], replace=False, 
                               size=int(layer.shape[1]*layer.shape[0]*failure_percentage))
            layer[np.unravel_index(indices, layer.shape)] = 0
    
    return network_weights
