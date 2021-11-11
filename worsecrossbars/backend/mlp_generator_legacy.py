"""
"""


from tensorflow.keras.layers import Dense, GaussianNoise, Activation
from tensorflow.keras.models import Sequential


def mnist_mlp_1hl(neurons=None, model_name="MNIST_MLP_1HL", noise_variance=1):
    """
    """

    # This network architecture was implemented to have one hidden layer, with about the same
    # number of trainable parameters as the two-layered architecture. This was done to ensure that
    # variability in fault simulation results was indeeed due to the number of layers being altered,
    # rather than to a different number of weights being implemented.
    # 89,050 parameters

    if neurons is None:
        neurons = [112]

    if not isinstance(neurons, list):
        raise TypeError("\"neurons\" should be a list object.")

    if len(neurons) != 1:
        raise ValueError("\"neurons\" list should have the same length as the number of layers \
                         being instantiated.")

    model = Sequential(name=model_name)

    if noise_variance:

        model.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        model.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,),
                        name=f"{model_name}_L1"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return model


def mnist_mlp_2hl(neurons=None, model_name="MNIST_MLP_2HL", noise_variance=1):
    """
    """

    # This is the network architecture employed in the *Simulation of Inference Accuracy Using
    # Realistic RRAM Devices* paper. It consists of a feed-forward multilayer perceptron with 784
    # input neurons (encoding pixel intensities for 28 × 28 pixel MNIST images), two 100-neuron
    # hidden layers, and 10 output neurons (each corresponding to one of the ten digits). The first
    # three layers employ a sigmoid activation function, whilst the output layer makes use of a
    # softmax activation function (which means a cross-entropy error function is then used
    # throughout the learning task). All 60,000 MNIST training images were employed, divided into
    # training and validation sets in a 3:1 ratio, as described in the aforementioned paper.
    # 89,610 parameters

    if neurons is None:
        neurons = [100, 100]

    if not isinstance(neurons, list):
        raise TypeError("\"neurons\" should be a list object.")

    if len(neurons) != 2:
        raise ValueError("\"neurons\" list should have the same length as the number of layers \
                         being instantiated.")

    model = Sequential(name=model_name)

    if noise_variance:

        model.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(neurons[1], name=f"{model_name}_L2"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        model.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,),
                        name=f"{model_name}_L1"))
        model.add(Dense(neurons[1], activation="sigmoid", name=f"{model_name}_L2"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return model


def mnist_mlp_3hl(neurons=None, model_name="MNIST_MLP_3HL", noise_variance=1):
    """
    """

    # This network architecture was implemented to have three hidden layers, with about the same
    # number of trainable parameters as the two-layered architecture. This was done to ensure that
    # variability in fault simulation results was indeeed due to the number of layers being altered,
    # rather than to a different number of weights being implemented.
    # 89,375 parameters

    if neurons is None:
        neurons = [90, 95, 95]

    if not isinstance(neurons, list):
        raise TypeError("\"neurons\" should be a list object.")

    if len(neurons) != 3:
        raise ValueError("\"neurons\" list should have the same length as the number of layers \
                         being instantiated.")

    model = Sequential(name=model_name)

    if noise_variance:

        model.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(neurons[1], name=f"{model_name}_L2"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(neurons[2], name=f"{model_name}_L3"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        model.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,),
                        name=f"{model_name}_L1"))
        model.add(Dense(neurons[1], activation="sigmoid", name=f"{model_name}_L2"))
        model.add(Dense(neurons[2], activation="sigmoid", name=f"{model_name}_L3"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return model


def mnist_mlp_4hl(neurons=None, model_name="MNIST_MLP_4HL", noise_variance=1):
    """
    """

    # This network architecture was implemented to have four hidden layers, with about the same
    # number of trainable parameters as the two-layered architecture. This was done to ensure that
    # variability in fault simulation results was indeeed due to the number of layers being altered,
    # rather than to a different number of weights being implemented.
    # 89,515 parameters

    if neurons is None:
        neurons = [85, 85, 85, 85]

    if not isinstance(neurons, list):
        raise TypeError("\"neurons\" should be a list object.")

    if len(neurons) != 4:
        raise ValueError("\"neurons\" list should have the same length as the number of layers \
                         being instantiated.")

    model = Sequential(name=model_name)

    if noise_variance:

        model.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(neurons[1], name=f"{model_name}_L2"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(neurons[2], name=f"{model_name}_L3"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(neurons[3], name=f"{model_name}_L4"))
        model.add(GaussianNoise(noise_variance))
        model.add(Activation("sigmoid"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        model.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,),
                        name=f"{model_name}_L1"))
        model.add(Dense(neurons[1], activation="sigmoid", name=f"{model_name}_L2"))
        model.add(Dense(neurons[2], activation="sigmoid", name=f"{model_name}_L3"))
        model.add(Dense(neurons[3], activation="sigmoid", name=f"{model_name}_L4"))
        model.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return model
