from tensorflow.keras.layers import Dense, GaussianNoise, Activation
from tensorflow.keras.models import Sequential


def MNIST_MLP_1HL(neurons=[112], noise=False, noise_variance=1, model_name="MNIST_MLP_1HL"):
    """
    """

    # This network architecture was implemented to have one hidden layer, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.
    # 89,050 parameters

    if len(neurons) != 1:
        raise ValueError("\"neurons\" list should have the same length as the number of layers being instantiated.")

    MNIST_MLP_1HL = Sequential(name=model_name)

    if noise:

        MNIST_MLP_1HL.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_1HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_1HL.add(Activation("sigmoid"))
        MNIST_MLP_1HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        MNIST_MLP_1HL.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_1HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return MNIST_MLP_1HL


def MNIST_MLP_2HL(neurons=[100, 100], noise=False, noise_variance=1, model_name="MNIST_MLP_2HL"):
    """
    """

    # This is the network architecture employed in the *Simulation of Inference Accuracy Using Realistic RRAM Devices*
    # paper. It consists of a feed-forward multilayer perceptron with 784 input neurons (encoding pixel intensities
    # for 28 × 28 pixel MNIST images), two 100-neuron hidden layers, and 10 output neurons (each corresponding to one
    # of the ten digits). The first three layers employ a sigmoid activation function, whilst the output layer makes
    # use of a softmax activation function (which means a cross-entropy error function is then used throughout the
    # learning task). All 60,000 MNIST training images were employed, divided into training and validation sets in a
    # 3:1 ratio, as described in the aforementioned paper.
    # 89,610 parameters

    if len(neurons) != 2:
        raise ValueError("\"neurons\" list should have the same length as the number of layers being instantiated.")

    MNIST_MLP_2HL = Sequential(name=model_name)

    if noise:

        MNIST_MLP_2HL.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_2HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_2HL.add(Activation("sigmoid"))
        MNIST_MLP_2HL.add(Dense(neurons[1], name=f"{model_name}_L2"))
        MNIST_MLP_2HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_2HL.add(Activation("sigmoid"))
        MNIST_MLP_2HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        MNIST_MLP_2HL.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_2HL.add(Dense(neurons[1], activation="sigmoid", name=f"{model_name}_L2"))
        MNIST_MLP_2HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return MNIST_MLP_2HL


def MNIST_MLP_3HL(neurons=[90, 95, 95], noise=False, noise_variance=1, model_name="MNIST_MLP_3HL"):
    """
    """

    # This network architecture was implemented to have three hidden layers, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.
    # 89,375 parameters

    if len(neurons) != 3:
        raise ValueError("\"neurons\" list should have the same length as the number of layers being instantiated.")

    MNIST_MLP_3HL = Sequential(name=model_name)

    if noise:

        MNIST_MLP_3HL.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_3HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_3HL.add(Activation("sigmoid"))
        MNIST_MLP_3HL.add(Dense(neurons[1], name=f"{model_name}_L2"))
        MNIST_MLP_3HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_3HL.add(Activation("sigmoid"))
        MNIST_MLP_3HL.add(Dense(neurons[2], name=f"{model_name}_L3"))
        MNIST_MLP_3HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_3HL.add(Activation("sigmoid"))
        MNIST_MLP_3HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        MNIST_MLP_3HL.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_3HL.add(Dense(neurons[1], activation="sigmoid", name=f"{model_name}_L2"))
        MNIST_MLP_3HL.add(Dense(neurons[2], activation="sigmoid", name=f"{model_name}_L3"))
        MNIST_MLP_3HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return MNIST_MLP_3HL


def MNIST_MLP_4HL(neurons=[85, 85, 85, 85], noise=False, noise_variance=1, model_name="MNIST_MLP_4HL"):
    """
    """

    # This network architecture was implemented to have four hidden layers, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.
    # 89,515 parameters

    if len(neurons) != 4:
        raise ValueError("\"neurons\" list should have the same length as the number of layers being instantiated.")

    MNIST_MLP_4HL = Sequential(name=model_name)

    if noise:

        MNIST_MLP_4HL.add(Dense(neurons[0], input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_4HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_4HL.add(Activation("sigmoid"))
        MNIST_MLP_4HL.add(Dense(neurons[1], name=f"{model_name}_L2"))
        MNIST_MLP_4HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_4HL.add(Activation("sigmoid"))
        MNIST_MLP_4HL.add(Dense(neurons[2], name=f"{model_name}_L3"))
        MNIST_MLP_4HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_4HL.add(Activation("sigmoid"))
        MNIST_MLP_4HL.add(Dense(neurons[3], name=f"{model_name}_L4"))
        MNIST_MLP_4HL.add(GaussianNoise(noise_variance))
        MNIST_MLP_4HL.add(Activation("sigmoid"))
        MNIST_MLP_4HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    else:

        MNIST_MLP_4HL.add(Dense(neurons[0], activation="sigmoid", input_shape=(784,), name=f"{model_name}_L1"))
        MNIST_MLP_4HL.add(Dense(neurons[1], activation="sigmoid", name=f"{model_name}_L2"))
        MNIST_MLP_4HL.add(Dense(neurons[2], activation="sigmoid", name=f"{model_name}_L3"))
        MNIST_MLP_4HL.add(Dense(neurons[3], activation="sigmoid", name=f"{model_name}_L4"))
        MNIST_MLP_4HL.add(Dense(10, activation="softmax", name=f"{model_name}_OL"))

    return MNIST_MLP_4HL
