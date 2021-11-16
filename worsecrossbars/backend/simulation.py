"""
simulation:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""

import copy
import gc
import logging
import numpy as np
from worsecrossbars.backend.weight_mapping import choose_extremes
from worsecrossbars.backend.weight_mapping import create_weight_interval
from worsecrossbars.backend.weight_mapping import discretise_weights
from worsecrossbars.backend.mlp_generator import mnist_mlp
from worsecrossbars.backend.mlp_trainer import train_mlp


def weight_alterations(network_weights, fault_type, failure_percentage, extremes_list):
    """
    This function takes in and modifies network weights to simulate the effect of faulty memristive
    devices being used in the physical implementation of the ANN.

    It should be noted that only synapse parameters (i.e. weights, not neuron biases) are being
    altered. This is achieved by only modifying even-numbered layers, given that, in a
    densely-connected MLP built with Keras, odd-numbered layers contain neuron biases.

    Args:
      network_weights: Array containing the weights as outputted by the training functions.
      fault_type: String, indicates which fault is being simulated.
      failure_percentage: Positive integer/float, percentage of devices affected by the fault.
      extremes_list: List containing minimum and maximum weight values in a given layer.

    altered_weights:
      Array containing the ANN weights altered to simulate the effect of the desired percentage of
      devices being affected by the specified fault.
    """

    if isinstance(failure_percentage, int):
        failure_percentage = float(failure_percentage)

    if not isinstance(failure_percentage, float) or failure_percentage < 0:
        raise ValueError("\"failure_percentage\" argument should be a positive real number.")

    altered_weights = copy.deepcopy(network_weights)

    for count, layer in enumerate(altered_weights):

        if count % 2 == 0:
            if fault_type == "STUCKZERO":
                fault_value = 0
            elif fault_type == "STUCKHRS":
                fault_value = extremes_list[count][0]
            else:
                fault_value = extremes_list[count][1]

            indices = np.random.choice(layer.shape[1]*layer.shape[0], replace=False,
                                       size=int(layer.shape[1]*layer.shape[0]*failure_percentage))

            # Creating a sign mask to ensure that devices stuck at HRS/LRS retain the correct sign
            # (i.e. that the associated weights remain negative if they were negative)
            signs_mask = np.sign(layer)

            layer[np.unravel_index(indices, layer.shape)] = fault_value * \
                signs_mask[np.unravel_index(indices, layer.shape)]

    return altered_weights


def fault_simulation(percentages_array, weights, network_model, dataset, simulation_parameters):
    """
    This function runs a fault simulation with the given parameters, and thus constitutes the
    computational core of the package. Each simulation is run "number_simulations" times, to average
    out stochastic variability in the final results.

    Args:
      percentages_array: Array containing the various percentages of faulty devices the user wants
        to simulate.
      weights: Array containing trained weights that are to be altered to simulate faults.
      network_model: Keras model containing the network topology being simulated.
      dataset: MNIST test dataset, used to calculate inference accuracy.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.

    accuracies:
      Array containing the ANN's inference accuracy at each percentage of faulty devices.
    """

    extremes_list = choose_extremes(weights, simulation_parameters["HRS_LRS_ratio"],
                                    simulation_parameters["excluded_weights_proportion"])
    weight_intervals = create_weight_interval(extremes_list,
                                              simulation_parameters["number_conductance_levels"])
    weights = discretise_weights(weights, weight_intervals)

    accuracies = np.zeros(len(percentages_array))

    for _ in range(simulation_parameters["number_simulations"]):

        accuracies_list = []

        for percentage in percentages_array:
            altered_weights = weight_alterations(weights, simulation_parameters["fault_type"],
                                                 percentage, extremes_list)

            # The "set_weights" function sets the ANN's weights to the values specified in the
            # list of arrays "altered_weights"
            network_model.set_weights(altered_weights)
            accuracies_list.append(network_model.evaluate(dataset[1][0],
                                   dataset[1][1], verbose=0)[1])

        accuracies += np.array(accuracies_list)
        gc.collect()

    accuracies /= simulation_parameters["number_simulations"]

    return accuracies


def train_models(dataset, simulation_parameters, epochs, batch_size):
    """
    This function trains the generated Keras models on the MNIST dataset with the given parameters.

    Args:
      dataset: MNIST dataset used to train the models.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.
      epochs: Positive integer, number of epochs over which the models will be trained.
      batch_size: Positive integer, size of batches that will be used to train the models.

    weights_list:
      List of arrays containing the weights of each of the "number_ANNs" trained networks.

    histories_list:
      List of Keras history dictionaries for each of the "number_ANNs" trained networks.
    """

    number_anns = simulation_parameters["number_ANNs"]
    number_hidden_layers = simulation_parameters["number_hidden_layers"]
    noise_variance = simulation_parameters["noise_variance"]

    weights_list = []
    histories_list = []

    for model_number in range(0, int(number_anns)):

        model = mnist_mlp(number_hidden_layers, noise_variance=noise_variance)
        mlp_weights, mlp_history, *_ = train_mlp(dataset, model, epochs, batch_size)
        weights_list.append(mlp_weights)
        histories_list.append(mlp_history)

        gc.collect()

        logging.info(f"[{number_hidden_layers}HL_{number_anns}ANNs_{noise_variance}NV]" +
                     f" Trained model {model_number+1} of {number_anns}")

    return weights_list, histories_list


def training_validation_metrics(histories_list):
    """
    This function calculates training and validation metrics by averaging the data generated during
    model training and stored in the Keras histories dictionaries returned by the train_models
    function.

    Args:
      histories_list: List containing Keras models' training histories, as output by the fit
        method run by the worsecrossbars.backend.mlp_trainer.train_mlp function.

    training_accuracy_values:
      Array containing the training accuracy values.

    validation_accuracy_values:
      Array containing the validation accuracy values.

    training_loss_values:
      Array containing the training loss values.

    validation_loss_values:
      Array containing the validation loss values.
    """

    training_accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
    validation_accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
    training_loss_values = np.zeros(len(histories_list[0].history["accuracy"]))
    validation_loss_values = np.zeros(len(histories_list[0].history["accuracy"]))

    for history in histories_list:

        history_dict = history.history
        training_accuracy_values += np.array(history_dict["accuracy"])
        validation_accuracy_values += np.array(history_dict["val_accuracy"])
        training_loss_values += np.array(history_dict["loss"])
        validation_loss_values += np.array(history_dict["val_loss"])

    training_accuracy_values /= len(histories_list)
    validation_accuracy_values /= len(histories_list)
    training_loss_values /= len(histories_list)
    validation_loss_values /= len(histories_list)

    return training_accuracy_values, validation_accuracy_values, \
           training_loss_values, validation_loss_values


def run_simulation(weights_list, percentages_array, dataset, simulation_parameters):
    """
    This function runs the main simulation. This entails running one full fault_simulation for each
    of the "number_ANNs" networks trained above, so that the accuracies resulting from each can be
    averaged together to reduce the influence of stochastic variability.

    Args:
      weights_list:
      percentages_array: Array containing the various percentages of faulty devices the user wants
        to simulate.
      dataset: MNIST test dataset, used to calculate inference accuracy.
      simulation_parameters: Python dictionary (loaded from a JSON file) containing all parameters
        needed in the simulation, including fault_type, HRS_LRS_ratio, excluded_weights_proportion,
        number_conductance_levels, number_simulations.

    accuracies_array:
      Array containing the average of all accuracies obtained for each of the "number_ANNs"
      networks (each itself run "number_simulations" times), for a total of "number_ANNs" *
      "number_simulations" datapoints averaged together to obtain each value stored in the final
      array.
    """

    number_anns = simulation_parameters["number_ANNs"]
    number_hidden_layers = simulation_parameters["number_hidden_layers"]
    noise_variance = simulation_parameters["noise_variance"]

    model = mnist_mlp(number_hidden_layers, noise_variance=noise_variance)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    accuracies_array = np.zeros((len(weights_list), len(percentages_array)))

    for count, weights in enumerate(weights_list):

        accuracies_array[count] = fault_simulation(percentages_array, weights, model, dataset,
                                                   simulation_parameters)

        logging.info(f"[{number_hidden_layers}HL_{number_anns}ANNs_{noise_variance}NV]" +
                     f" Simulated model {count+1} of {number_anns}.")
        gc.collect()

    return np.mean(accuracies_array, axis=0, dtype=np.float64)
