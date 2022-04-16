"""nonidealities:
A backend module used to simulate various memristive nonidealities.
"""
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp


class StuckAtValue:
    """This class ..."""

    def __init__(self, value: float, probability: float = 0.0) -> None:
        self.value = value
        self.probability = probability
        self.is_linearity_preserving = True

    def __repr__(self) -> str:

        return f"StuckAtValue (Value: {self.value}; Probability: {self.probability*100}%)"

    def alter_conductances(self, conductances: tf.Tensor, **kwargs) -> tf.Tensor:
        """"""

        # Unpacking kwargs
        prob_mask = kwargs.get("prob_mask", None)

        if prob_mask is None:
            # Creating a mask of bools to alter a given percentage of conductance values
            mask = (
                tf.random.uniform(conductances.shape, 0, 1, dtype=tf.dtypes.float64)
                < self.probability
            )
        else:
            mask = prob_mask < self.probability

        altered_conductances = tf.where(mask, self.value, conductances)

        return altered_conductances

    def update(self, probability: float) -> None:
        """"""

        self.probability = probability
        return None


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
            self.g_on = kwargs.get("g_on", None)
            self.g_off = kwargs.get("g_off", None)
            if self.num_of_weights is None or self.g_off is None or self.g_on is None:
                raise ValueError(
                    "g_on, g_off, and num_of_weights must be supplied if no distrib is given!"
                )
            self.distrib = (
                tf.random.uniform([self.num_of_weights], minval=self.g_off, maxval=self.g_on)
                .numpy()
                .tolist()
            )

    def __repr__(self) -> str:

        return f"StuckDistribution (Distrib: {self.distrib}; Probability: {self.probability*100}%)"

    def alter_conductances(self, conductances: tf.Tensor, **kwargs) -> tf.Tensor:
        """"""

        # Unpacking kwargs
        prob_mask = kwargs.get("prob_mask", None)
        indices: tf.Variable = kwargs.get("indices")

        if prob_mask is None:
            # Creating a mask of bools to alter a given percentage of conductance values
            mask = (
                tf.random.uniform(conductances.shape, 0, 1, dtype=tf.dtypes.float64)
                < self.probability
            )
        else:
            mask = prob_mask < self.probability

        count = tf.math.count_nonzero(
            tf.math.equal(indices, tf.constant(-1, shape=conductances.shape, dtype=tf.dtypes.int32))
        )
        if count > 0:
            indices.assign(
                tf.random.uniform(
                    conductances.shape, minval=0, maxval=self.num_of_weights, dtype=tf.int32
                )
            )

        altered_conductances = conductances
        for index, level in enumerate(self.distrib):
            altered_conductances = tf.where(tf.equal(indices, index), level, altered_conductances)

        altered_conductances = tf.where(mask, altered_conductances, conductances)

        return altered_conductances

    def update(self, probability: float) -> None:
        """"""

        self.probability = probability
        return None


class D2DVariability:
    """This class ..."""

    def __init__(self, g_off: float, g_on: float, on_std: float, off_std: float) -> None:
        self.is_linearity_preserving = True
        self.g_off = g_off
        self.g_on = g_on
        self.on_std = on_std
        self.off_std = off_std

    def __repr__(self) -> str:

        return f"D2DVariability (On_std: {self.on_std}; Off_std: {self.off_std})"

    def alter_conductances(self, conductances: tf.Tensor, **kwargs) -> tf.Tensor:

        resistances = 1 / conductances
        resistance_on = 1 / self.g_on
        resistance_off = 1 / self.g_off

        log_std = tfp.math.interp_regular_1d_grid(
            resistances, resistance_on, resistance_off, [self.on_std, self.off_std]
        )

        res_squared = tf.math.pow(resistances, 2)
        res_var = res_squared * (tf.math.exp(tf.math.pow(log_std, 2)) - 1.0)
        log_mu = tf.math.log(res_squared / tf.math.sqrt(res_squared + res_var))

        resistances = tfp.distributions.LogNormal(log_mu, log_std, validate_args=True).sample()

        altered_conductances = 1 / resistances

        return altered_conductances


class IVNonlinear:
    """This class ..."""

    def __init__(self, v_ref: float, avg_gamma: float, std_gamma: float) -> None:
        self.v_ref = v_ref
        self.avg_gamma = avg_gamma
        self.std_gamma = std_gamma
        self.k_v = 2 * self.v_ref
        self.is_linearity_preserving = False

    def __repr__(self) -> str:

        return (
            f"IVNonlinear (v_ref: {self.v_ref}; Avg_gamma: {self.avg_gamma}; "
            + "Std_gamma: {self.std_gamma})"
        )

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
        ohmic_currents = voltage_signs * self.v_ref * tf.expand_dims(conductances, 0)
        v_v_ref_ratio = tf.expand_dims(tf.abs(voltages) / self.v_ref, -1)

        log_gammas = tf.experimental.numpy.log2(gammas)

        individual_currents = ohmic_currents * v_v_ref_ratio ** log_gammas
        currents = tf.math.reduce_sum(individual_currents, axis=1)

        return currents, individual_currents
