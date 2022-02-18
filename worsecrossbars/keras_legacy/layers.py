import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from worsecrossbars.keras_legacy import mapping

class MemristiveFullyConnected (layers.Layer):

    def __init__(
        self,
        neurons_in: int,
        neurons_out: int,
        G_off: float,
        G_on: float,
        k_V: float,
        **kwargs
    ) -> None:

        # Assigning positional arguments to the layer
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.G_off = G_off
        self.G_on = G_on
        self.k_V = k_V

        # Unpacking kwargs
        self.nonidealities = kwargs.get("nonidealities", [])
        self.regulariser = kwargs.get("regulariser", "L1")
        self.mapping_rule = kwargs.get("mapping_rule", "lowest")
        self.uses_double_weights = kwargs.get("uses_double_weights", False)

        super(MemristiveFullyConnected, self).__init__()

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.neurons_out)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.neurons_out)
    
    def build(self, input_shape):

        self.w = self.add_weight(
            shape = (self.neurons_in, self.neurons_out),
            initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = (1/np.sqrt(self.neurons_in))),
            name = "weights",
            trainable = True
        )

        self.b = self.add_weight(
            shape = (self.neurons_out, ),
            initializer = tf.keras.initializers.Constant(value = 0.0),
            name = "biases",
            trainable = True
        )

    def call(self, x, mask=None):

        if not tf.keras.backend.learning_phase():
            return tf.tensordot(x, self.w, axes=1) + self.b

        inputs = tf.concat([x, tf.ones([tf.shape(x)[0], 1])], 1)

        # Calculating layers outputs
        self.out = self.memristive_outputs(inputs, self.combine_weights())

        return self.out

    def combine_weights(self):
        """"""

        bias = tf.expand_dims(self.b, 0)
        combined_weights = tf.concat([self.w, bias], 0)
        return combined_weights

    def memristive_outputs(self, x, weights):
        """"""

        # Converting neuronal inputs to voltages
        voltages = self.k_V * x

        # Mapping network weights to conductances
        conductances, max_weight = mapping.weights_to_conductances(
            weights, self.G_off, self.G_on, self.mapping_rule
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
            if tf.keras.backend.learning_phase():
                currents = tf.tensordot(voltages, conductances, 1)
            else:
                individual_currents = tf.expand_dims(voltages, -1) * tf.expand_dims(conductances, 0)
                currents = tf.math.reduce_sum(individual_currents, 1)

        total_currents = currents[:, 0::2] - currents[:, 1::2]
        k_cond = (self.G_on - self.G_off) / max_weight
        y_disturbed = total_currents / (self.k_V * k_cond)

        return y_disturbed
    
