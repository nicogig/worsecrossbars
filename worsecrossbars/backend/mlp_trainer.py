"""Contains the create_datasets and train_mlp functions"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def create_datasets(train_validation_ratio=3):
    """
    This function creates traning and validation datasets based on the MNIST digit database,
    according to the given training/validation split.

    Args:
      train_validation_ratio: Positive integer/float, ratio between size of training and validation
        datasets, indicating that the training dataset is "train_validation_ratio" times bigger
        than the validation dataset.

    train_validation_dataset:
      Tuple containing the training and validation images and labels.

    test_dataset:
      Tuple containing the testing images and labels.
    """

    # Dataset download
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Data reshaping
    mlp_train_images = train_images.reshape((60000, 28 * 28)).astype("float32")/255
    mlp_test_images = test_images.reshape((10000, 28 * 28)).astype("float32")/255
    mlp_train_labels = to_categorical(train_labels)
    mlp_test_labels = to_categorical(test_labels)

    # Creating a validation set
    validation_size = round(mlp_train_images.shape[0]/(train_validation_ratio+1))
    mlp_validation_data = mlp_train_images[:validation_size]
    mlp_partial_train = mlp_train_images[validation_size:]
    mlp_validation_labels = mlp_train_labels[:validation_size]
    mlp_partial_labels = mlp_train_labels[validation_size:]

    # Packaging datasets into tuples
    train_validation_dataset = (mlp_validation_data, mlp_validation_labels,
                                mlp_partial_train, mlp_partial_labels)
    test_dataset = (mlp_test_images, mlp_test_labels)

    return train_validation_dataset, test_dataset


def train_mlp(dataset, model, epochs=10, batch_size=100):
    """
    This function trains a given Keras model on the dataset provided to it.

    Args:
      dataset: Tuple of tuples, containing training, validation and testing images and labels, as
        provided by the create_datasets() function defined above.
      model: Keras model which is to be trained
      epochs: Positive integer, number of epochs used in training.
      batch_size: Positive integer, number of batches used in training.

    mlp_weights:
      List containing the parameters associated with each layer of the ANN. mlp_weights[0] comprises
      the weights related to the synapses between layer 1 and layer 2, mlp_weights[1] contains the
      bias terms for the neurons in layer 2, and so forth.

    mlp_history:
      Keras history object, containing information regarding network performance at different
      epochs.

    mlp_test_loss:
      Final test loss

    mlp_test_acc:
      Final test accuracy
    """

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("\"epochs\" argument should be an integer greater than 1.")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("\"batch_size\" argument should be an integer greater than 1.")

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # Training with validation test
    mlp_history = model.fit(dataset[0][2], dataset[0][3], epochs=epochs, batch_size=batch_size,
                            validation_data=(dataset[0][0], dataset[0][1]), verbose=0)
    mlp_test_loss, mlp_test_acc = model.evaluate(dataset[1][0], dataset[1][1], verbose=0)

    # Extracting network weights
    mlp_weights = model.get_weights()

    return mlp_weights, mlp_history, mlp_test_loss, mlp_test_acc
