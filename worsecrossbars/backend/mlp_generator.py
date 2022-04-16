"""mlp_generator:
A backend module used to create a Keras model for a densely connected MLP with a given topology.
"""
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Sequential

from worsecrossbars.backend.layers import MemristiveFullyConnected


def mnist_mlp(
    g_off: float,
    g_on: float,
    k_v: float,
    nonidealities: list = None,
    number_hidden_layers: int = 2,
    neurons: List[int] = None,
    model_name: str = "",
    noise_variance: float = 0.0,
    **kwargs,
) -> Model:
    """This function returns a Keras model set up to be trained to recognise digits from the MNIST
    dataset (784 input neurons, 10 softmax output neurons).

    The default network architecture (2HLs) consists of a feed-forward multilayer perceptron with
    784 input neurons (encoding pixel intensities for 28 x 28 pixel MNIST images), two 100-neuron
    hidden layers, and 10 output neurons (each corresponding to one of the ten digits). The first
    three layers employ a sigmoid activation function, whilst the output layer makes use of a
    softmax activation function (which means a cross-entropy error function is then used
    throughout the learning task). All 60,000 MNIST training images were employed, divided into
    training and validation sets in a 3:1 ratio, as described in the aforementioned paper.

    For the one-layer, three-layers and four-layers topologies, the number of neurons in each
    hidden layer was tweaked so as to produce a final network with about the same number of
    trainable parameters as the original, two-layers ANN. This was done to ensure that variability
    in fault simulation results was indeeed due to the number of layers being altered, rather than
    to a different number of weights being implemented.

    The function also gives the user the option to add GaussianNoise layers with a specific variance
    between hidden layers during training. This is done to increase the network's generalisation
    power, as well as to increase resilience to faulty memristive devices.

    Args:
      g_off: The off-state conductance of the memristor.
      g_on: The on-state conductance of the memristor.
      k_v: Memristive reference voltage.
      nonidealities: A list of non-idealities to be applied to the network.
      number_hidden_layers: Integer comprised between 1 and 4, number of hidden layers instantiated
        as part of the model.
      neurons: List of length number_hidden_layers, contains the number of neurons to be created in
        each densely-connected layer.
      model_name: String, name of the Keras model.
      noise_variance: Positive integer/float, variance of the GaussianNoise layers instantiated
        during training to boost network performance.

    Returns:
      model: Keras model object, contaning the desired topology.
    """

    # Unpacking kwargs
    horovod = kwargs.get("horovod", False)
    conductance_drifting = kwargs.get("conductance_drifting", True)
    optimiser = kwargs.get("optimiser", "adam")
    model_size = kwargs.get("model_size", "small")
    uses_double_weights = kwargs.get("double_weights", True)

    default_neurons = {
        "big": {1: [112], 2: [100, 100], 3: [91, 91, 91], 4: [85, 85, 85, 85]},
        "regular": {1: [53], 2: [50, 50], 3: [47, 47, 47], 4: [45, 45, 45, 45]},
        "small": {1: [26], 2: [25, 25], 3: [24, 24, 24], 4: [24, 24, 24, 24]},
        "tiny": {1: [12], 2: [12, 12], 3: [12, 12, 12], 4: [12, 12, 12, 12]},
    }

    if not isinstance(model_name, str):
        raise ValueError('"model_name" argument should be a string object.')

    if not isinstance(model_size, str) or model_size not in ["big", "regular", "small", "tiny"]:
        raise ValueError(
            '"model_size" argument should be a string equal to "big", "regular", "small" or "tiny".'
        )

    selected_default_neurons = default_neurons[model_size]

    # Setting default argument values
    if neurons is None:
        neurons = selected_default_neurons[number_hidden_layers]
    if model_name == "":
        model_name = f"MNIST_MLP_{number_hidden_layers}HL"
    if nonidealities is None:
        nonidealities = []

    if not isinstance(neurons, list) or len(neurons) != number_hidden_layers:
        raise ValueError(
            '"neurons" argument should be a list object with the same length as '
            + "the number of layers being instantiated."
        )

    model = Sequential(name=model_name)

    # Creating first hidden layer
    model.add(
        MemristiveFullyConnected(
            784,
            neurons[0],
            g_off,
            g_on,
            k_v,
            nonidealities=nonidealities,
            conductance_drifting=conductance_drifting,
            uses_double_weights=uses_double_weights,
        )
    )

    if noise_variance:
        model.add(GaussianNoise(noise_variance))

    model.add(Activation("sigmoid"))

    # Creating other hidden layers
    for layer_index, _ in enumerate(neurons[1:]):

        model.add(
            MemristiveFullyConnected(
                neurons[layer_index],
                neurons[layer_index + 1],
                g_off,
                g_on,
                k_v,
                nonidealities=nonidealities,
                conductance_drifting=conductance_drifting,
                uses_double_weights=uses_double_weights,
            )
        )

        if noise_variance:
            model.add(GaussianNoise(noise_variance))

        model.add(Activation("sigmoid"))

    # Creating output layer
    model.add(
        MemristiveFullyConnected(
            neurons[-1],
            10,
            g_off,
            g_on,
            k_v,
            nonidealities=nonidealities,
            conductance_drifting=conductance_drifting,
            uses_double_weights=uses_double_weights,
        )
    )
    model.add(Activation("softmax"))

    if horovod:
        import horovod.tensorflow as hvd

        if optimiser == "adam":
            opt = tf.keras.optimizers.Adam(0.001 * hvd.size())
        elif optimiser == "sgd":
            opt = tf.keras.optimizers.SGD(0.01 * hvd.size())
        elif optimiser == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(0.001 * hvd.size())
        opt = hvd.DistributedOptimizer(opt)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        if optimiser == "adam":
            opt = tf.keras.optimizers.Adam()
        elif optimiser == "sgd":
            opt = tf.keras.optimizers.SGD()
        elif optimiser == "rmsprop":
            opt = tf.keras.optimizers.RMSprop()
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def get_number_parameters(model_size: str) -> List[int]:

    model_sizes = []

    for num_hl in range(1, 5):

        model = mnist_mlp(0, 0, 0, number_hidden_layers=num_hl, model_size=model_size)
        model.build((1, 784))
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        # non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        model_sizes.append(trainable_count)

    return model_sizes
