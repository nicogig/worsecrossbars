"""layers:
A backend module dedicated to the creation of custom synaptic layers for TensorFlow.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from worsecrossbars.backend import mapping


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
        self.noise = torch.tensor(0, dtype=torch.float, device=device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class MemristiveLinear(nn.Module):
    """This class ..."""

    def __init__(
        self,
        neurons_in: int,
        neurons_out: int,
        G_off: float,
        G_on: float,
        k_V: float,
        device: torch.device,
        **kwargs
    ) -> None:

        super().__init__()

        # Assigning positional arguments to the layer
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.G_off = G_off
        self.G_on = G_on
        self.k_V = k_V
        self.device = device

        # Unpacking kwargs
        self.nonidealities = kwargs.get("nonidealities", [])
        self.regulariser = kwargs.get("regulariser", "L1")
        self.mapping_rule = kwargs.get("mapping_rule", "lowest")
        self.uses_double_weights = kwargs.get("uses_double_weights", False)

        # Building layer
        self.build()

    def build(self):
        """"""

        # Initialising weights according to a normal distribution with mean 0 and standard deviation
        # equal to 1 / np.sqrt(self.neurons_in)
        self.w = Parameter(
            torch.normal(0.0, 1 / np.sqrt(self.neurons_in), (self.neurons_in, self.neurons_out)).to(
                self.device
            )
        )

        # Initialising layer biases to zero
        self.b = Parameter(torch.zeros(self.neurons_out, device=self.device))

    def combine_weights(self):
        """"""

        bias = torch.unsqueeze(self.b, dim=0)
        combined_weights = torch.cat([self.w, bias], 0).to(self.device)
        return combined_weights

    def memristive_outputs(self, x, weights):
        """"""

        # Converting neuronal inputs to voltages
        voltages = self.k_V * x

        # Mapping network weights to conductances
        conductances, max_weight = mapping.weights_to_conductances(
            weights, self.G_off, self.G_on, self.device, self.mapping_rule
        )

        # Applying linearity-preserving nonidealities
        for nonideality in self.nonidealities:
            if nonideality.is_linearity_preserving:
                conductances = nonideality.alter_conductances(conductances)

        # Applying linearity-non-preserving nonidealities
        currents, individual_currents = None, None
        for nonideality in self.nonidealities:
            if not nonideality.is_linearity_preserving:
                currents, individual_currents = nonideality.calc_currents(voltages, conductances)

        # If no linearity-non-preserving nonideality is present, calculate output currents in an
        # ideal fashion
        if currents is None or individual_currents is None:
            if self.training:
                currents = torch.tensordot(voltages, conductances, dims=1).to(self.device)
            else:
                individual_currents = torch.unsqueeze(voltages, dim=-1).to(
                    self.device
                ) * torch.unsqueeze(conductances, dim=0).to(self.device)
                currents = torch.sum(individual_currents, dim=1).to(self.device)

        total_currents = currents[:, 0::2] - currents[:, 1::2]
        k_cond = (self.G_on - self.G_off) / max_weight
        y_disturbed = total_currents / (self.k_V * k_cond)

        return y_disturbed

    def forward(self, x):
        """"""

        inputs = torch.cat([x, torch.ones([x.size()[0], 1], device=self.device)], 1).to(self.device)

        # Calculating layers outputs
        self.out = self.memristive_outputs(inputs, self.combine_weights())

        return self.out
