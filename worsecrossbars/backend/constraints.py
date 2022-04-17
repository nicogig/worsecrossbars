import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class DiscreteWeights(Constraint):
    """"""

    def __init__(self, conductance_levels, non_neg=True) -> None:

        self.conductance_levels = conductance_levels
        self.non_neg = non_neg

    def __call__(self, weights):

        indices = tf.raw_ops.Bucketize(input=weights, boundaries=self.conductance_levels)

        # Bucketize has an extra interval in [last_element, +inf)
        # The following brings it back to an acceptable element
        # -nicogig
        mask = indices > len(self.conductance_levels) - 1
        indices = tf.where(mask, len(self.conductance_levels) - 1, indices)

        for index, cond_level in enumerate(self.conductance_levels):
            weights = tf.where(tf.equal(indices, index), cond_level, weights)

        if self.non_neg:
            print(weights)
            return weights * tf.cast(tf.math.greater_equal(weights, 0.0), weights.dtype)

        return weights
