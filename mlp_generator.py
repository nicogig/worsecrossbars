from tensorflow.keras import layers
from tensorflow.keras import models

def one_layer():
    pass



def two_layers():

    MNIST_MLP = models.Sequential(name="MNIST_MLP")
    MNIST_MLP.add(layers.Dense(100, activation="sigmoid", input_shape=(784,), name="MNIST_MLP_L1"))
    MNIST_MLP.add(layers.Dense(100, activation="sigmoid", name="MNIST_MLP_L2"))
    MNIST_MLP.add(layers.Dense(10, activation="softmax", name="MNIST_MLP_OL"))
    return MNIST_MLP



def three_layers():
    pass



def four_layers():
    pass