import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from worsecrossbars.backend import mapping


class MemristiveFullyConnected(layers.Layer):
    """This class ..."""

    def __init__(
        self,
        neurons_in: int,
        neurons_out: int,
        G_off: float,
        G_on: float,
        k_V: float,
        is_training: bool = False,
        **kwargs
    ) -> None:

        # Assigning positional arguments to the layer
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.G_off = G_off
        self.G_on = G_on
        self.k_V = k_V
        self.is_training = is_training

        # Unpacking kwargs
        self.nonidealities = kwargs.get("nonidealities", [])
        self.regulariser = kwargs.get("regulariser", "L1")
        self.mapping_rule = kwargs.get("mapping_rule", "lowest")
        self.uses_double_weights = kwargs.get("uses_double_weights", True)

        self.prob_mask = None

        super().__init__()

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.neurons_out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.neurons_out)

    def build(self, input_shape):

        stdv = 1 / np.sqrt(self.neurons_in)

        if self.uses_double_weights:

            self.w_pos = self.add_weight(
                shape=(self.neurons_in, self.neurons_out),
                initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
                name="weights_pos",
                trainable=True,
                constraint=tf.keras.constraints.NonNeg(),
            )

            self.w_neg = self.add_weight(
                shape=(self.neurons_in, self.neurons_out),
                initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=stdv),
                name="weights_neg",
                trainable=True,
                constraint=tf.keras.constraints.NonNeg(),
            )

            self.b_pos = self.add_weight(
                shape=(self.neurons_out,),
                initializer=tf.keras.initializers.Constant(value=0.5),
                name="biases_pos",
                trainable=True,
                constraint=tf.keras.constraints.NonNeg(),
            )

            self.b_neg = self.add_weight(
                shape=(self.neurons_out,),
                initializer=tf.keras.initializers.Constant(value=0.5),
                name="biases_neg",
                trainable=True,
                constraint=tf.keras.constraints.NonNeg(),
            )

        else:

            self.w = self.add_weight(
                shape=(self.neurons_in, self.neurons_out),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv),
                name="weights",
                trainable=True,
            )

            self.b = self.add_weight(
                shape=(self.neurons_out,),
                initializer=tf.keras.initializers.Constant(value=0.0),
                name="biases",
                trainable=True,
            )

        self.built = True

    def call(self, x, mask=None):
        """"""

        if not self.uses_double_weights and self.nonidealities == []:
            return tf.tensordot(x, self.w, axes=1) + self.b

        inputs = tf.concat([x, tf.ones([tf.shape(x)[0], 1])], 1)

        # Calculating layers outputs
        self.out = self.memristive_outputs(inputs, self.combine_weights())

        return self.out

    def combine_weights(self):
        """"""

        if self.uses_double_weights:

            pos_biases = tf.expand_dims(self.b_pos, 0)
            neg_biases = tf.expand_dims(self.b_neg, 0)

            comb_pos = tf.concat([self.w_pos, pos_biases], 0)
            comb_neg = tf.concat([self.w_neg, neg_biases], 0)

            combined_weights = tf.reshape(
                tf.concat([comb_pos[..., tf.newaxis], comb_neg[..., tf.newaxis]], -1),
                [tf.shape(comb_pos)[0], -1],
            )
        else:

            bias = tf.expand_dims(self.b, 0)
            combined_weights = tf.concat([self.w, bias], 0)

        return combined_weights

    @tf.function
    def memristive_outputs(self, x, weights):
        """"""

        # Converting neuronal inputs to voltages
        voltages = self.k_V * x

        # Mapping network weights to conductances
        if self.uses_double_weights:
            conductances, max_weight = mapping.double_weights_to_conductances(
                weights, self.G_off, self.G_on
            )
        else:
            conductances, max_weight = mapping.weights_to_conductances(
                weights, self.G_off, self.G_on, self.mapping_rule
            )
        
        if self.prob_mask is None:
            self.prob_mask = tf.random.uniform(conductances.shape, 0, 1, dtype=tf.dtypes.float64)

        # Either this is not working correctly
        # or it is extremely detrimental to the network.
        # Maybe bucketization should not be something the network is aware of? (i.e. do on device)

        # cond_levels = weights_manipulation.gen_conductance_level(self.G_off, self.G_on, 100, 1000)
        # conductances = weights_manipulation.bucketize_weights(conductances, cond_levels)

        # Applying linearity-preserving nonidealities
        for nonideality in self.nonidealities:
            if nonideality.is_linearity_preserving:
                # Gen a probability mask for the current layer if one has not been generated yet.
                conductances = nonideality.alter_conductances(conductances, self.prob_mask)

        # Applying linearity-non-preserving nonidealities
        currents, individual_currents = None, None
        for nonideality in self.nonidealities:
            if not nonideality.is_linearity_preserving:
                currents, individual_currents = nonideality.calc_currents(voltages, conductances)

        # If no linearity-non-preserving nonideality is present, calculate output currents in an
        # ideal fashion
        if currents is None or individual_currents is None:
            if self.is_training:
                currents = tf.tensordot(voltages, conductances, 1)
            else:
                individual_currents = tf.expand_dims(voltages, -1) * tf.expand_dims(conductances, 0)
                currents = tf.math.reduce_sum(individual_currents, 1)

        total_currents = currents[:, 0::2] - currents[:, 1::2]
        k_cond = (self.G_on - self.G_off) / max_weight
        y_disturbed = total_currents / (self.k_V * k_cond)

        return y_disturbed
