from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from black import out
from layers import GaussianNoise


class MultiLayerPerceptron(nn.Module):
    """
    A Multi-Layer Perceptron network built using the PyTorch framework.

    Args:
        num_hidden_layers: Integer comprised between 1 and 4, number of hidden layers instantiated as
            part of the model.
        device: The device the simulation will be performed on. This MUST be a torch.device() object!
            ATTN: This is NOT the Memristor device!
        neurons: List of length num_hidden_layers, contains the number of neurons to be created in
            each densely-connected layer.
        model_name: String, name of the Keras model.
        noise_variance: Positive integer/float, variance of the GaussianNoise layers instantiated
            during training to boost network performance.

    """

    def __init__(
        self,
        num_hidden_layers: int,
        device,
        neurons: List[int] = None,
        model_name: str = "",
        noise_variance: float = 0.0,
    ) -> None:

        super().__init__()

        default_neurons = {1: [112], 2: [100, 100], 3: [90, 95, 95], 4: [85, 85, 85, 85]}
        self.noise_variance = noise_variance

        if neurons is None:
            neurons = default_neurons[num_hidden_layers]
        if model_name == "":
            self.model_name = f"MNIST_MLP_{num_hidden_layers}HL"
        else:
            self.model_name = model_name

        self.dense_1 = nn.Linear(in_features=784 * 1, out_features=neurons[0])
        if noise_variance:
            self.noise_1 = GaussianNoise(device, sigma=noise_variance)

        self.hidden_layers = nn.ModuleList()
        for _, neuron in enumerate(neurons[1:]):
            self.central_layers.append(nn.Linear(in_features=neuron, out_features=neuron))
            if noise_variance:
                self.central_layers.append(GaussianNoise(device, sigma=noise_variance))

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.dense_1(x)

        if self.noise_variance:
            x = self.noise_1(x)

        for layer in self.hidden_layers:
            x = layer(x)

        output = F.softmax(x, dim=1)
        return output
