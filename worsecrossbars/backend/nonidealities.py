"""nonidealities:
A backend module used to simulate various memristive nonidealities.
"""
from typing import Tuple

import tensorflow as tf


class StuckAtValue:
    """This class ..."""

    def __init__(self, value: float, probability: float) -> None:
        self.value = value
        self.probability = probability
        self.is_linearity_preserving = True

    def alter_conductances(self, conductances: tf.Tensor) -> tf.Tensor:
        """A method to alter conductances stored in a PyTorch Tensor.

        Args:
            conductances: A PyTorch tensor containing memristive conductances.

        Returns:
            altered_conductances: A PyTorch tensor containing the altered memristive conductances.
        """

        # Creating a mask of bools to alter a given percentage of conductance values
        mask = (
            tf.random.uniform(conductances.shape, 0, 1, dtype=tf.dtypes.float64) < self.probability
        )
        altered_conductances = tf.where(mask, self.value, conductances)

        return altered_conductances


class StuckDistribution:
    """This class ..."""

    def __init__(self) -> None:
        pass


class D2DVariability:
    """This class ..."""

    def __init__(self) -> None:
        pass


class IVNonlinear:
    """This class ..."""

    def __init__(self, V_ref: float, avg_gamma: float, std_gamma: float) -> None:
        self.V_ref = V_ref
        self.avg_gamma = avg_gamma
        self.std_gamma = std_gamma
        self.k_V = 2 * self.V_ref
        self.is_linearity_preserving = False

    def calc_currents(
        self, voltages: tf.Tensor, conductances: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """This function implements Equation 9 from "Nonideality-Aware Training for Accurate and
        Robust Low-Power Memristive Neural Networks" in order to compute output currents in a MCBA
        affected by IV nonlinearities. The lowest possible value assigned to nonlinearity parameter
        gamma is 2, in line with what is discussed in the paper.

        Args:
          voltages:
          conductances:

        Returns:
          currents:
          individual_currents:
        """

        # Generating a truncated normal distribution (with a low value of 2) describing the possible
        # values taken on by the nonlinearity parameter gamma, which we shall sample from in order
        # to assign a nonlinearity parameter to each memristor in the crossbar array.
        gammas = tf.random.truncated_normal(conductances.shape, self.avg_gamma, self.std_gamma)

        voltage_signs = tf.expand_dims(tf.sign(voltages), -1)
        ohmic_currents = voltage_signs * self.V_ref * tf.expand_dims(conductances, 0)
        v_v_ref_ratio = tf.expand_dims(tf.abs(voltages) / self.V_ref, -1)

        log_gammas = tf.experimental.numpy.log2(gammas)

        individual_currents = ohmic_currents * v_v_ref_ratio ** log_gammas
        currents = tf.math.reduce_sum(individual_currents, axis=1)

        return currents, individual_currents
