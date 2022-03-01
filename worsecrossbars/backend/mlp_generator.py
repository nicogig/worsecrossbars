"""mlp_generator:
A backend module used to create a Keras model for a densely connected MLP with a given topology.
"""
from typing import List

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Sequential
import tensorflow as tf

from worsecrossbars.backend.layers import MemristiveFullyConnected


def mnist_mlp(
    G_off: float,
    G_on: float,
    k_V: float,
    nonidealities: list = [],
    number_hidden_layers: int = 2,
    neurons: List[int] = None,
    model_name: str = "",
    noise_variance: float = 0.0,
    debug: bool = False,
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
      G_off:
      G_on:
      k_V:
      nonidealities:
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

    default_neurons = {1: [112], 2: [100, 100], 3: [90, 95, 95], 4: [85, 85, 85, 85]}

    # Setting default argument values
    if neurons is None:
        neurons = default_neurons[number_hidden_layers]
    if model_name == "":
        model_name = f"MNIST_MLP_{number_hidden_layers}HL"

    if debug:
        number_hidden_layers = 1
        neurons = [25]

    if not isinstance(neurons, list) or len(neurons) != number_hidden_layers:
        raise ValueError(
            '"neurons" argument should be a list object with the same length as '
            + "the number of layers being instantiated."
        )

    if not isinstance(model_name, str):
        raise ValueError('"model_name" argument should be a string object.')

    model = Sequential(name=model_name)

    # Creating first hidden layer
    model.add(
        MemristiveFullyConnected(784, neurons[0], G_off, G_on, k_V, nonidealities=nonidealities)
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
                G_off,
                G_on,
                k_V,
                nonidealities=nonidealities,
            )
        )

        if noise_variance:
            model.add(GaussianNoise(noise_variance))

        model.add(Activation("sigmoid"))

    # Creating output layer
    model.add(
        MemristiveFullyConnected(neurons[-1], 10, G_off, G_on, k_V, nonidealities=nonidealities)
    )
    model.add(Activation("softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
