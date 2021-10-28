from tensorflow.keras import layers
from tensorflow.keras import models

def one_layer():
    """
    """

    # This network architecture was implemented to have one hidden layer, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.
    
    # 89,050 parameters
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(112, activation="sigmoid", input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def one_layer_noise():
    """
    """

    # This network architecture was implemented to have one hidden layer, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.
    
    # 89,050 parameters
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(112, input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def two_layers():
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
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(100, activation="sigmoid", input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.Dense(100, activation="sigmoid", name="MNIST_MLP_L2"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def two_layers_noise():
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
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(100, input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(100, name="MNIST_MLP_L2"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def three_layers():
    """
    """

    # This network architecture was implemented to have three hidden layers, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.

    # 89,375 parameters
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(90, activation="sigmoid", input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.Dense(95, activation="sigmoid", name="MNIST_MLP_L2"))
    MNIST_MLP.add(layers.Dense(95, activation="sigmoid", name="MNIST_MLP_L3"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def three_layers_noise():
    """
    """

    # This network architecture was implemented to have three hidden layers, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.

    # 89,375 parameters
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(90, input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(95, name="MNIST_MLP_L2"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(95, name="MNIST_MLP_L3"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def four_layers():
    """
    """

    # This network architecture was implemented to have four hidden layers, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.

    # 89,515 parameters
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(85, activation="sigmoid", input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.Dense(85, activation="sigmoid", name="MNIST_MLP_L2"))
    MNIST_MLP.add(layers.Dense(85, activation="sigmoid", name="MNIST_MLP_L3"))
    MNIST_MLP.add(layers.Dense(85, activation="sigmoid", name="MNIST_MLP_L4"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def four_layers_noise():
    """
    """

    # This network architecture was implemented to have four hidden layers, with about the same number of trainable
    # parameters as the two-layered architecture. This was done to ensure that variability in fault simulation results
    # was indeeed due to the number of layers being altered, rather than to a different number of weights being implemented.

    # 89,515 parameters
    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(85, input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(85, name="MNIST_MLP_L2"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(85, name="MNIST_MLP_L3"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(85, name="MNIST_MLP_L4"))
    MNIST_MLP.add(layers.GaussianNoise(1))
    MNIST_MLP.add(layers.Activation("sigmoid"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP
