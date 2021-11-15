"""
mlp_generator_pytorch:
A backend module used to create a PyTorch model for a densely connected MLP with a given topology.
"""

import numpy as np
import torch
from torch import nn


class MNIST_MLP(nn.Module):
    """
    This class implements a PyTorch model set up to be trained to recognise digits from the MNIST
    dataset (784 input neurons, 10 softmax output neurons).

    The network architecture corresponds to that employed in the "Simulation of Inference Accuracy
    Using Realistic RRAM Devices" paper. It consists of a feed-forward multilayer perceptron with
    784 input neurons (encoding pixel intensities for 28 × 28 pixel MNIST images), two 100-neuron
    hidden layers, and 10 output neurons (each corresponding to one of the ten digits). The first
    three layers employ a sigmoid activation function, whilst the output layer makes use of a
    softmax activation function (which means a cross-entropy error function is then used
    throughout the learning task). All 60,000 MNIST training images were employed, divided into
    training and validation sets in a 3:1 ratio, as described in the aforementioned paper.

    For the one-layer, three-layers and four-layers topologies, the number of neurons in each
    hidden layer was tweaked so as to produce a final network with about the same number of
    trainable parameters as the original, two-layers ANN. This was done to ensure that variability
    in fault simulation results was indeeed due to the number of layers being altered, rather than
    to a different number of weights being implemented.

    The function also gives the user the option to add GaussianNoise layers with a specific variance
    between hidden layers during training. This is done to increase the network's generalisation
    power, as well as to increase resilience to faulty memristive devices.

    Args:
      number_hidden_layers: Integer comprised between 1 and 4, number of hidden layers instantiated
        as part of the model.
      hidden_layer_sizes: NumPy ndarray of size number_hidden_layers, contains the number of neurons
        to be instantiated in each densely-connected layer.
      noise_variance: Positive integer/float, variance of the GaussianNoise layers instantiated
        during training to boost network performance.
    """

    def __init__(self, number_hidden_layers, hidden_layer_sizes=None):
        """
        """

        super().__init__()

        default_neurons = {1: np.array([112]), 2: np.array([100, 100]), 3: np.array([90, 95, 95]),
                           4: np.array([85, 85, 85, 85])}

        # Setting default argument values
        if hidden_layer_sizes is None:
            hidden_layer_sizes = default_neurons[number_hidden_layers]

        if not isinstance(hidden_layer_sizes, np.ndarray) or \
           hidden_layer_sizes.size != number_hidden_layers:
            raise ValueError("\"hidden_layer_sizes\" argument should be a NumPy ndarray object " +
                             "with the same size as the number of layers being instantiated.")

        # Activation layers
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

        # Hidden layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(784, hidden_layer_sizes[0]))

        for index in range(number_hidden_layers-1):
            self.hidden.append(nn.Linear(hidden_layer_sizes[index], hidden_layer_sizes[index+1]))

        # Output layer
        self.output = nn.Linear(hidden_layer_sizes[-1], 10)


    def forward(self, x):
        """
        """

        # Flattening input image
        x = x.view(-1, 784)

        for layer in self.hidden:
            x = torch.sigmoid(layer(x))

        output = torch.softmax(self.output(x), dim=1)

        return output