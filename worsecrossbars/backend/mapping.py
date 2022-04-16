"""mapping:
A backend module used to map memristive conductances to synaptic layers, and vice versa.
"""
import tensorflow as tf


@tf.function
def double_weights_to_conductances(weights: tf.Tensor, g_off: float, g_on: float) -> tf.Tensor:
    """"""

    maximum_weight = tf.math.reduce_max(tf.math.abs(weights))
    scaling_fact_weight = (g_on - g_off) / maximum_weight

    conductance = (scaling_fact_weight * weights) + g_off

    return conductance, maximum_weight


@tf.function
def conductances_to_double_weights(
    conductances: tf.Tensor, g_off: float, g_on: float, max_weight: float
):
    return (conductances - g_off) / ((g_on - g_off) / max_weight)


@tf.function
def weights_to_conductances(
    weights: tf.Tensor,
    g_off: float,
    g_on: float,
    mapping_style: str = "lowest",
) -> tf.Tensor:
    """This functions maps Tensorflow weights onto conductances.

    Args:
        weights: The original layer of weights as a Tensor.
        g_off: A float representing the conductance of the devices in their OFF state.
        g_on: A float representing the conductance of the devices in their ON state.
        mapping_style: A string, either `lowest` or `average`.
            Represents the mapping style to be utilised, where
            `lowest` instructs the function to map according to the lowest possible conductance, and
            `average` instructs the function to map the weights around an average conductance.

    Returns:
        conductance_layer: A Tensorflow tensor representing the mapped conductances, shape `m x 2n`.
        maximum_weight: The max weight in the original weights layer.
    """

    maximum_weight = tf.math.reduce_max(tf.math.abs(weights))
    scaling_fact_weight = (g_on - g_off) / maximum_weight
    effective_cond = scaling_fact_weight * weights

    if mapping_style == "lowest":

        # Map according to lowest possible conductance
        cond_pos = tf.math.maximum(effective_cond, 0.0) + g_off
        cond_neg = -tf.math.maximum(effective_cond, 0.0) + g_off

    elif mapping_style == "average":

        # Map around the average conductance
        avg_cond = (g_on + g_off) / 2
        cond_pos, cond_neg = avg_cond + 0.5 * effective_cond, avg_cond - 0.5 * effective_cond

    else:

        raise ValueError('"mapping_style" parameter is not valid.')

    conductance_layer = tf.reshape(
        tf.concat([cond_pos[..., tf.newaxis], cond_neg[..., tf.newaxis]], axis=-1),
        [tf.shape(cond_pos)[0], -1],
    )

    return conductance_layer, maximum_weight
