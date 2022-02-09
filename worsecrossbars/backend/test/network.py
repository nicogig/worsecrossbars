from typing import List
from black import out
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GaussianNoise

class MultiLayerPerceptron(nn.Module):
    """
    ... todo.
    """
    
    def __init__(self, 
        num_hidden_layers: int,
        device,
        neurons: List[int] = None,
        model_name: str = "",
        noise_variance: float = 0.0) -> None:
        
        super(MultiLayerPerceptron, self).__init__()

        default_neurons = {1: [112], 2: [100, 100], 3: [90, 95, 95], 4: [85, 85, 85, 85]}
        self.noise_variance = noise_variance

        if neurons is None:
            neurons = default_neurons[num_hidden_layers]
        if model_name == "":
            self.model_name = f"MNIST_MLP_{num_hidden_layers}HL"
        else:
            self.model_name = model_name
        
        self.dense_1 = nn.Linear(
                in_features=784*1,
                out_features=neurons[0]
            )
        if noise_variance:
            self.noise_1 = GaussianNoise(device, sigma=noise_variance)
        
        self.central_layers = []
        for _, neuron in enumerate(neurons[1:]):
            self.central_layers.append(
                nn.Linear(
                    in_features=neuron,
                    out_features=neuron
                ))
            if noise_variance:
                self.central_layers.append(
                    GaussianNoise(device, sigma=noise_variance)
                )
    
    def forward(self, x):
        x = self.dense_1(x)
        
        if self.noise_variance:
            x = self.noise_1(x)
        
        for layer in self.central_layers:
            x = layer(x)
        
        output = F.softmax(x, dim=1)
        return output