"""nonidealities:
A backend module used to simulate various memristive nonidealities.
"""
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class LinearityPreserving(ABC):
    @abstractmethod
    def alter_G(self, conductances: torch.Tensor) -> torch.Tensor:
        """A method to disturb conductances in a PyTorch Tensor.

        Args:
            conductances: A PyTorch tensor containing memristive conductances.

        Returns:
            altered_conductances
        """


class LinearityNonPreserving(ABC):
    @abstractmethod
    def calc_I(self, voltages: torch.Tensor, conductances: torch.Tensor) -> torch.Tensor:
        """
        TODO: pipo
        """


class StuckAtValue(LinearityPreserving):
    """This class ..."""

    def __init__(self, value: float, probability: float, label: str) -> None:
        self.value = value
        self.probability = probability
        self.label = f"{label}: {value::.2g}, {probability::.2g}"

    def alter_G(self, conductances: torch.Tensor) -> torch.Tensor:

        # Creating a mask of bools to alter a given percentage of conductance values
        mask = torch.rand(conductances.shape, dtype=torch.float64) < self.probability
        conductances = torch.where(mask, self.value, conductances)

        return conductances


class StuckDistribution:
    def __init__(self) -> None:
        pass


class D2DVariability:
    def __init__(self) -> None:
        pass


class IVNonlinear(LinearityNonPreserving):
    """This class ..."""

    def __init__(self, V_ref: float, avg_gamma: float, std_gamma: float, label: str) -> None:
        self.V_ref = V_ref
        self.avg_gamma = avg_gamma
        self.std_gamma = std_gamma
        self.label = f"{label}: {avg_gamma:.2g}, {std_gamma:.2g}"
        self.k_V = 2 * self.V_ref

    def calc_I(self, voltages, conductances):
        """This function implements Equation 9 from "Nonideality-Aware Training for Accurate and
        Robust Low-Power Memristive Neural Networks" in order to compute output currents in a MCBA
        affected by IV nonlinearities. The lowest possible value assigned to nonlinearity parameter
        gamma is 2, in line with what is discussed in the paper.

        Args:
          voltages:
          conductances:

        Returns:
          currents:

        """

        # Generating a truncated normal distribution (with a low value of 2) describing the possible
        # values taken on by the nonlinearity parameter gamma, which we shall sample from in order
        # to assign a nonlinearity parameter to each memristor in the crossbar array.

        gammas = nn.init.trunc_normal_(
            torch.zeros(conductances.get_shape().as_list()),
            mean=self.avg_gamma,
            std=self.std_gamma,
            a=2.0,
            b=np.inf,
        )
        # gamma_distr = tfp.distributions.TruncatedNormal(self.avg_gamma, self.std_gamma, 2.0, np.inf)
        # gammas = gamma_distr.sample(sample_shape=conductances.get_shape().as_list())

        v_sign_tensor = torch.unsqueeze(torch.sign(voltages), -1)
        ohmic_current = v_sign_tensor * self.V_ref * torch.unsqueeze(conductances, 0)
        v_v_ref_ratio = torch.unsqueeze(torch.abs(voltages) / self.V_ref, -1)

        log_sampled_n = torch.log(gammas)
        exponent = log_sampled_n / torch.log(torch.full([1], 2, dtype=log_sampled_n.dtype))

        individual_I = ohmic_current * v_v_ref_ratio ** exponent
        I = torch.sum(individual_I, dim=1)

        return I, individual_I
