"""mlp_trainer:
A backend module used to create dataset and train a Keras model on them.
"""
import math
from typing import List
from typing import Tuple

from numpy import ndarray
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from worsecrossbars.backend import weights_manipulation
from worsecrossbars.backend.layers import MemristiveFullyConnected
from worsecrossbars.backend import mapping


def get_dataset(
    dataset: str,
    training_validation_ratio: float,
) -> Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]:
    """This function creates traning and validation datasets based on either the MNIST digit
    database, or on the CIFAR-10 image database, according to the given training/validation split.

    Args:
      dataset: String indicating whether the data should be MNIST or CIFAR-10.
      training_validation_ratio: Positive integer/float, ratio between size of training and
        validation datasets, indicating that the training dataset is "training_validation_ratio"
        times bigger than the validation dataset.

    Returns:
      validation_data: Array containing validation data.
      validation_labels: Array containing validation labels.
      training_data: Array containing training data.
      training_labels: Array containing training labels.
      test_data: Array containing test data.
      test_labels: Array containing test labels.
    """

    if isinstance(training_validation_ratio, int):
        training_validation_ratio = float(training_validation_ratio)

    if not isinstance(training_validation_ratio, float) or training_validation_ratio < 0:
        raise ValueError('"training_validation_ratio" argument should be a positive real number.')

    # Dataset download, normalisation and reshaping
    if dataset == "mnist":
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_data = train_images.reshape((60000, 28 * 28)).astype("float32") / 255.0
        test_data = test_images.reshape((10000, 28 * 28)).astype("float32") / 255.0
    elif dataset == "cifar10":
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_data = train_images.astype("float32") / 255.0
        test_data = test_images.astype("float32") / 255.0
    else:
        raise ValueError('"dataset" parameter should be a string equal to "mnist" or "cifar10".')

    # Casting labels to categorical form (one-hot encoding)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Creating a validation set
    validation_size = round(train_data.shape[0] / (training_validation_ratio + 1))
    validation_data = train_data[:validation_size]
    training_data = train_data[validation_size:]
    validation_labels = train_labels[:validation_size]
    training_labels = train_labels[validation_size:]

    return (validation_data, validation_labels, training_data, training_labels), (
        test_data,
        test_labels,
    )


def train_mlp(
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    model: Model,
    epochs: int,
    batch_size: int,
    horovod: bool = False,
    **kwargs
) -> Tuple[List[ndarray], History]:
    """This function trains a given Keras model on the dataset provided to it.

    Args:
      dataset: Tuple of tuples, containing training, validation and testing images and labels, as
        provided by the dataset functions defined above.
      model: Keras model which is to be trained
      epochs: Positive integer, number of epochs used in training.
      batch_size: Positive integer, number of batches used in training.
      **kwargs: Valid keyword arguments are listed below
        - discretise: Boolean, specifies whether the network weights should be discrete or not.
        - hrs_lrs_ratio:
        - number_conductance_levels:
        - excluded_weights_proportion:

    Returns:
      mlp_weights: List containing the parameters associated with each layer of the ANN.
        mlp_weights[0] comprises the weights related to the synapses between layer 1 and layer 2,
        mlp_weights[1] contains the bias terms for the neurons in layer 2, and so forth.
      mlp_history: Keras history object, containing information regarding network performance at
        different epochs.
      pre_discretisation_accuracy:
    """

    # kwargs unpacking
    discretise = kwargs.get("discretise", False)  # Default should be conductance_drifting
    hrs_lrs_ratio = kwargs.get("hrs_lrs_ratio", 5)
    number_conductance_levels = kwargs.get("number_conductance_levels", 10)
    excluded_weights_proportion = kwargs.get("excluded_weights_proportion", 0.015)
    nonidealities = kwargs.get("nonidealities", [])

    if not isinstance(model, Model):
        raise ValueError('"model" argument should be a Keras model object.')

    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError('"epochs" argument should be an integer greater than 1.')

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('"batch_size" argument should be an integer greater than 1.')

    if not isinstance(discretise, bool):
        raise ValueError('"discretise" argument should be a boolean.')

    if not isinstance(hrs_lrs_ratio, float):
        raise ValueError('"hrs_lrs_ratio" argument should be int/float.')

    if not isinstance(number_conductance_levels, int):
        raise ValueError('"number_conductance_levels" argument should be an integer.')

    if (
        not isinstance(excluded_weights_proportion, float)
        and not 0 <= excluded_weights_proportion <= 1
    ):
        raise ValueError(
            '"excluded_weights_proportion" argument should be a float value between 0 and 1.'
        )

    # Training with validation
    compute_steps_per_epoch = lambda x: int(math.ceil(1.0 * x / batch_size))

    if horovod:
        import horovod.tensorflow as hvd

        scaled_lr = 0.001 * hvd.size()
        callbacks = [
            hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.keras.callbacks.MetricAverageCallback(),
            hvd.keras.callbacks.LearningRateWarmupCallback(
                initial_lr=scaled_lr, warmup_epochs=3, verbose=1
            ),
        ]
        if hvd.rank() == 0:
            verbose = 2
        else:
            verbose = 0
        steps = (compute_steps_per_epoch(len(dataset[0][2])) * 2) // hvd.size()
    else:
        callbacks = []
        verbose = 2
        steps = compute_steps_per_epoch(len(dataset[0][2]))

    model.is_training = True
    mlp_history = model.fit(
        dataset[0][2],
        dataset[0][3],
        epochs=epochs,
        steps_per_epoch=steps,
        batch_size=batch_size,
        validation_data=(dataset[0][0], dataset[0][1]),
        callbacks=callbacks,
        verbose=verbose,
    )
    model.is_training = False
    model.run_eagerly = False

    pre_discretisation_accuracy = model.evaluate(dataset[1][0], dataset[1][1])[1]

    # If discrete weights are being used, the the bucketize_weights_layer function is employed.
    if discretise:

        for layer in model.layers:

            if isinstance(layer, MemristiveFullyConnected):

                if layer.uses_double_weights:

                    discrete_w_pos = weights_manipulation.bucketize_weights_layer(
                        layer.w_pos.read_value(),
                        hrs_lrs_ratio,
                        number_conductance_levels,
                        excluded_weights_proportion,
                    )
                    discrete_w_neg = weights_manipulation.bucketize_weights_layer(
                        layer.w_neg.read_value(),
                        hrs_lrs_ratio,
                        number_conductance_levels,
                        excluded_weights_proportion,
                    )

                    layer.w_pos.assign(discrete_w_pos)
                    layer.w_neg.assign(discrete_w_neg)

                else:

                    discrete_w = weights_manipulation.bucketize_weights_layer(
                        layer.w.read_value(),
                        hrs_lrs_ratio,
                        number_conductance_levels,
                        excluded_weights_proportion,
                    )

                    layer.w.assign(discrete_w)

    # Extracting network weights
    mlp_weights = model.get_weights()

    return mlp_weights, mlp_history, pre_discretisation_accuracy
