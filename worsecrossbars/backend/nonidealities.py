"""nonidealities:
A backend module used to simulate various memristive nonidealities.
"""
import copy
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp


class StuckAtValue:
    """This class ..."""

    def __init__(self, value: float, probability: float = 0.0) -> None:
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

    def __init__(self, probability: float = 0.0, distrib: list = None, **kwargs) -> None:

        self.probability = probability
        self.is_linearity_preserving = True
        if distrib is not None:
            self.distrib = distrib
            self.num_of_weights = len(distrib)
        else:
            self.num_of_weights = kwargs.get("num_of_weights", None)
            self.G_on = kwargs.get("G_on", None)
            self.G_off = kwargs.get("G_off", None)
            if self.num_of_weights is None or self.G_off is None or self.G_on is None:
                raise ValueError(
                    "G_on, G_off, and num_of_weights must be supplied if no distrib is given!"
                )
            self.distrib = (
                tf.random.uniform([self.num_of_weights], minval=self.G_off, maxval=self.G_on)
                .numpy()
                .tolist()
            )

    def alter_conductances(self, conductances: tf.Tensor) -> tf.Tensor:
        """ """
        mask = (
            tf.random.uniform(conductances.shape, 0, 1, dtype=tf.dtypes.float64) < self.probability
        )
        indices = tf.random.uniform(
            conductances.shape, minval=0, maxval=self.num_of_weights, dtype=tf.int32
        )
        altered_conds = copy.deepcopy(conductances)

        for index, level in enumerate(self.distrib):
            altered_conds = tf.where(tf.equal(indices, index), level, altered_conds)

        return tf.where(mask, altered_conds, conductances)


class D2DVariability:
    """This class ..."""

    def __init__(self, G_off: float, G_on: float, on_std: float, off_std: float) -> None:
        self.is_linearity_preserving = True
        self.G_off = G_off
        self.G_on = G_on
        self.on_std = on_std
        self.off_std = off_std

    def alter_conductances(self, conductances: tf.Tensor) -> tf.Tensor:

        resistances = 1 / conductances
        resistance_on = 1 / self.G_on
        resistance_off = 1 / self.G_off

        log_std = tfp.math.interp_regular_1d_grid(
            resistances, resistance_on, resistance_off, [self.on_std, self.off_std]
        )

        res_squared = tf.math.pow(resistances, 2)
        res_var = res_squared * (tf.math.exp(tf.math.pow(log_std, 2)) - 1.0)
        log_mu = tf.math.log(res_squared / tf.math.sqrt(res_squared + res_var))

        resistances = tfp.distributions.LogNormal(log_mu, log_std, validate_args=True).sample()

        return 1 / resistances


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
