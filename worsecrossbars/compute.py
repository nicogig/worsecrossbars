"""
compute.py (formerly MLP.py)
Worsecrossbars main module and entrypoint.
"""

import argparse
import sys
import signal
import gc
import platform
import pickle
from pathlib import Path
import numpy as np
from worsecrossbars.backend.MLP_generator import MNIST_MLP_1HL
from worsecrossbars.backend.MLP_generator import MNIST_MLP_2HL
from worsecrossbars.backend.MLP_generator import MNIST_MLP_3HL
from worsecrossbars.backend.MLP_generator import MNIST_MLP_4HL
from worsecrossbars.backend.MLP_trainer import dataset_creation, train_MLP
from worsecrossbars.backend.fault_simulation import run_simulation
from worsecrossbars.utilities import initial_setup, json_handlers
from worsecrossbars.utilities import io_operations
from worsecrossbars.utilities.logging_module import Logging
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier
from worsecrossbars.utilities.dropbox_upload import DropboxUpload

def stop_handler(signum, _):
    """
    A stop signal handler.
    """
    if command_line_args.log:
        log.write("Simulation terminated unexpectedly. Got signal" + \
            f" {signal.Signals(signum).name}.\nEnding.\n")
        log.write(special="abruptend")
    if command_line_args.teams:
        teams.send_message(f"Simulation ({number_hidden_layers}" + \
            f" {HIDDEN_LAYER}, fault type {fault_type})" + \
            f" terminated unexpectedly.\nGot signal {signal.Signals(signum).name}.\nEnding.", \
             title="Simulation ended", color="b90e0a")
    gc.collect()
    sys.exit(0)

def main(): ## too_many_statements, too_many_variables
    """
    main(command_line_args):
    The main function.
    """
    if command_line_args.dropbox:
        dbx = DropboxUpload(output_folder)

    mnist_dataset = dataset_creation()
    weights_list = []
    histories_list = []
    generator_functions = {1: MNIST_MLP_1HL, 2: MNIST_MLP_2HL, 3: MNIST_MLP_3HL, 4: MNIST_MLP_4HL}

    # Model definition and training,
    # repeated "number_ANNs" times to average out stochastic variancies
    for model_number in range(0, int(number_anns)):

        mnist_mlp = generator_functions[number_hidden_layers](noise=True,
        noise_variance=extracted_json["noise_variance"])
        mlp_weights, mlp_history, *_ = train_MLP(mnist_dataset,
                                                mnist_mlp,
                                                epochs=10,
                                                batch_size=100)
        weights_list.append(mlp_weights)
        histories_list.append(mlp_history)
        gc.collect()

        if command_line_args.log:
            log.write(string=f"Trained model {model_number+1} of {number_anns}")

    # Computing training and validation loss and accuracy
    # by averaging over all the models trained in the previous step
    if command_line_args.log:
        log.write(string="Done training. Computing loss and accuracy.")

    #epochs = range(1, len(histories_list[0].history["accuracy"]) + 1)
    accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
    validation_accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
    loss_values = np.zeros(len(histories_list[0].history["accuracy"]))
    validation_loss_values = np.zeros(len(histories_list[0].history["accuracy"]))

    for history in histories_list:

        history_dict = history.history
        accuracy_values += np.array(history_dict["accuracy"])
        validation_accuracy_values += np.array(history_dict["val_accuracy"])
        loss_values += np.array(history_dict["loss"])
        validation_loss_values += np.array(history_dict["val_loss"])

    accuracy_values /= len(histories_list)
    validation_accuracy_values /= len(histories_list)
    loss_values /= len(histories_list)
    validation_loss_values /= len(histories_list)

    # Saving training/validation data to file
    with open(str(Path.home().joinpath("worsecrossbars", "outputs",
               output_folder, "training_validation",
               f"training_validation_faultType{fault_type}_{number_hidden_layers}HL" + \
                f"_{noise_variance}NV.pickle")), "wb") as file:
        pickle.dump(
            (accuracy_values, validation_accuracy_values, loss_values, validation_loss_values),
            file)

    if command_line_args.log:
        log.write(string="Saved training and validation data.")

    # Running "args.number_simulations" simulations
    # for each of the "args.number_ANNs" networks trained above over the specified
    # range of faulty devices percentages
    mnist_mlp = generator_functions[number_hidden_layers](noise=True, noise_variance=noise_variance)
    mnist_mlp.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    percentages = np.arange(0, 1.01, 0.01)
    accuracies_array = np.zeros((len(weights_list), len(percentages)))
    fault_num = {"STUCK_ZERO":1, "STUCK_HIGH":2, "STUCK_LOW":3}

    for count, weights in enumerate(weights_list):

        accuracies_array[count] = run_simulation(percentages,
                                                weights,
                                                int(number_simulations),
                                                mnist_mlp,
                                                mnist_dataset,
                                                fault_num[fault_type],
                                                extracted_json["HRS_LRS_ratio"],
                                                extracted_json["number_of_conductance_levels"],
                                                extracted_json["excluded_weights_proportion"])
        gc.collect()
        if command_line_args.log:
            log.write(string=f"Simulated model {count+1} of {number_anns}.")

    # Averaging the results obtained for each of the 30 sets of weights
    accuracies = np.mean(accuracies_array, axis=0, dtype=np.float64)

    # Saving accuracies array to file
    with open(str(Path.home().joinpath("worsecrossbars", "outputs",
               output_folder, "accuracies",
               f"accuracies_faultType{fault_type}_{number_hidden_layers}HL" + \
                f"_{noise_variance}NV.pickle")), "wb") as file:
        pickle.dump((percentages, accuracies, fault_num[fault_type]), file)

    if command_line_args.log:
        log.write(special="end")

    if command_line_args.teams:
        teams.send_message(f"Finished script using parameters {number_hidden_layers}" + \
                            f" HL, {fault_type} fault type.",
                            "Finished execution", color="028a0f")
    if command_line_args.dropbox:
        dbx.upload()

if __name__ == "__main__":

    # Command line parser for input arguments
    parser=argparse.ArgumentParser()

    parser.add_argument("--setup", dest="setup", metavar="INITIAL_SETUP", \
         help="Run the inital setup", type=bool, default=False)
    parser.add_argument("--config", dest="config", metavar="CONFIG_FILE", \
         help="Provide the config file needed for simulations", type=str)
    parser.add_argument("-w", dest="wipe_current", metavar="WIPE_CURRENT", \
         help="Wipe the current output (or config)", type=bool, default=False)
    parser.add_argument("-l", dest="log", metavar="LOG", \
         help="Enable logging the output in a separate file", type=bool, default=True)
    parser.add_argument("-d", dest="dropbox", metavar="DROPBOX", \
         help="Enable Dropbox integration", type=bool, default=True)
    parser.add_argument("-t", dest="teams", metavar="MSTEAMS", \
         help="Enable MS Teams integration", type=bool, default=True)

    command_line_args = parser.parse_args()

    if command_line_args.setup:
        initial_setup.main_setup(command_line_args.wipe_current)
        sys.exit(0)
    else:
        # Get the JSON supplied, parse it, validate it against
        # a known schema.
        json_path = Path.cwd().joinpath(command_line_args.config)
        extracted_json = io_operations.read_external_json(str(json_path))
        json_handlers.validate_json(extracted_json)

        # Create User, Output folders.
        io_operations.user_folders()
        output_folder = io_operations.create_output_structure(extracted_json,
        command_line_args.wipe_current)

        # Extract Useful info from JSON Object
        number_hidden_layers = extracted_json["number_hidden_layers"]
        HIDDEN_LAYER = "hidden layer" if number_hidden_layers == 1 else "hidden layers"
        fault_type = extracted_json["fault_type"]
        number_anns = extracted_json["number_ANNs"]
        noise_variance = extracted_json["noise_variance"]
        number_simulations = extracted_json["number_simulations"]

        if command_line_args.log:
            log = Logging(extracted_json, output_folder)
            log.write(special="begin")
        if command_line_args.teams:
            teams = MSTeamsNotifier(io_operations.read_webhook())
            teams.send_message(f"Using parameters: {number_hidden_layers} {HIDDEN_LAYER}," + \
                f" fault type {fault_type}.", title="Started new simulation", color="028a0f")
        # Attach Signal Handler
        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        if platform.system() == "Darwin" or platform.system() == "Linux":
            signal.signal(signal.SIGHUP, stop_handler)
        main() # Goto Main
