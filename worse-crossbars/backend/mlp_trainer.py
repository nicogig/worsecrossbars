from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

def dataset_creation():
    """
    dataset_creation: Create the MNIST dataset used for training.
    Inputs:
        -   none.
    Outputs:
        -   A tuple of tuples containing the validation data, validation labels, partial training data, partial training labels, test data, test labels.
    """

    # Dataset download
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # print() # Prints a newline after the dataset download info

    # Data reshaping
    MLP_train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
    MLP_test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
    MLP_train_labels = to_categorical(train_labels)
    MLP_test_labels = to_categorical(test_labels)

    # Creating a validation set
    MLP_validation_data = MLP_train_images[:15000]
    MLP_partial_train = MLP_train_images[15000:]
    MLP_validation_labels = MLP_train_labels[:15000]
    MLP_partial_labels = MLP_train_labels[15000:]

    return (MLP_validation_data, MLP_validation_labels, MLP_partial_train, MLP_partial_labels), (MLP_test_images, MLP_test_labels)



def train_MLP(dataset, model, epochs=10, batch_size=100):
    """
    train_MLP:
    Train a Neural Network model on the MNIST dataset.
    Inputs:
        -   dataset: The MNIST dataset, as provided by the dataset_creation() function
        -   model: A Keras model.
        -   epochs: The number of epochs used in training. Default: 10.
        -   batch_size: The size of the batches used for training. Default: 100.
    Outputs:
        -   A tuple containing the weights, history, test loss, and test accuracy of the network.
    """

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # Training with validation test
    MLP_history = model.fit(dataset[0][2], dataset[0][3], epochs=epochs, batch_size=batch_size,
                validation_data=(dataset[0][0], dataset[0][1]),verbose=0)
    MLP_test_loss, MLP_test_acc = model.evaluate(dataset[1][0], dataset[1][1], verbose=0)

    # Extracting network weights for alterations carried out in the sections below.
    # MLP_weights is a list containing the parameters associated with each layer of the ANN. MLP_weights[0] comprises the weights related to
    # the 78,400 synapses between layer 1 and layer 2, MLP_weights[1] contains the bias terms for the 10 neurons in layer 2, and so forth
    MLP_weights = model.get_weights()

    return MLP_weights, MLP_history, MLP_test_loss, MLP_test_acc
