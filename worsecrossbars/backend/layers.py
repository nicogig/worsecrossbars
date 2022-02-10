"""layers:
A backend module dedicated to the creation of custom synaptic layers for TensorFlow.

Comment: Please do not touch this in any way, I'm still working on it!
- nicogig
"""
from dataclasses import dataclass
from typing import Any
from typing import List

import numpy as np
import torch
import torch.nn as nn
from backend import mapping
from backend import nonidealities
from torch.nn.parameter import Parameter


@dataclass
class MemristorLayerConfig:
    nonidealities: List[Any]
    is_regularized: bool
    G_off: float
    G_on: float
    k_V: float
    mapping_rule: str = "lowest"
    is_training: bool = True
    uses_double_weights: bool = False


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, device, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class LinearMemristorLayer(nn.Module):
    def __init__(self, neurons_in, neurons_out, config: MemristorLayerConfig) -> None:
        super().__init__()
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.config = config
        self.build()

    def build(self):
        std_dv = 1 / np.sqrt(self.neurons_in)
        self.w = Parameter(
            torch.normal(mean=0.0, std=std_dv, size=(self.neurons_in, self.neurons_out))
        )
        self.b = Parameter(torch.zeros(size=(self.neurons_out,)))

    def combined_weights(self):
        bias = torch.unsqueeze(self.b, dim=0)
        combined_weights = torch.cat([self.w, bias], 0)
        return combined_weights

    def memristive_outputs(self, x, weights):
        voltages = self.config.k_V * x
        conductances, max_weight = mapping.weights_to_conductances(
            weights, self.config.G_off, self.config.G_on, self.config.mapping_rule
        )
        for nonideality in self.config.nonidealities:
            if isinstance(nonideality, nonidealities.LinearityPreserving):
                conductances = nonideality.alter_G(conductances)

        i, i_individual = None, None
        for nonideality in self.config.nonidealities:
            if isinstance(nonideality, nonidealities.LinearityNonPreserving):
                i, i_individual = nonideality.calc_I(voltages, conductances)

        if i is None or i_individual is None:
            if self.config.is_training:
                i = torch.tensordot(voltages, conductances, dims=1)
            else:
                i_individual = torch.unsqueeze(voltages, dim=-1) * torch.unsqueeze(
                    conductances, dim=0
                )
                i = torch.sum(i_individual, dim=1)

        i_total = i[:, 0::2] - i[:, 1::2]
        k_cond = (self.config.G_on - self.config.G_off) / max_weight
        y_disturbed = i_total / (self.config.k_V * k_cond)

        return y_disturbed

    def forward(self, x):
        ones = torch.ones([x.size[0], 1])
        inputs = torch.cat([x, ones], 1)

        self.out = self.memristive_outputs(inputs, self.combined_weights())
        return self.out
