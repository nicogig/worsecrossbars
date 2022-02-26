import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class DiscreteWeights(Constraint):
    """This class ..."""

    def __init__(self, conductance_levels, non_neg=True) -> None:

        self.conductance_levels = conductance_levels
        self.non_neg = non_neg

    def __call__(self, w):

        indices = tf.raw_ops.Bucketize(input=w, boundaries=self.conductance_levels)

        # Bucketize has an extra interval in [last_element, +inf)
        # The following brings it back to an acceptable element
        # -nicogig
        mask = indices > len(self.conductance_levels) - 1
        indices = tf.where(mask, len(self.conductance_levels) - 1, indices)

        for index, cond_level in enumerate(self.conductance_levels):
            w = tf.where(tf.equal(indices, index), cond_level, w)

        if self.non_neg:
            print(w)
            return w * tf.cast(tf.math.greater_equal(w, 0.0), w.dtype)
        else:
            return w
