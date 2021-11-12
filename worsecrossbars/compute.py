"""
compute:
Worsecrossbars' main module and entrypoint.
"""

import argparse
import sys
import signal
import gc
import platform
import pickle
from pathlib import Path
import numpy as np
from worsecrossbars.backend.mlp_trainer import create_datasets
from worsecrossbars.backend.simulation import training_validation_metrics
from worsecrossbars.backend.simulation import train_models
from worsecrossbars.backend.simulation import run_simulation
from worsecrossbars.utilities.parameter_validator import validate_parameters
from worsecrossbars.utilities.initial_setup import main_setup
from worsecrossbars.utilities.json_handlers import validate_json
from worsecrossbars.utilities.io_operations import read_external_json
from worsecrossbars.utilities.io_operations import read_webhook
from worsecrossbars.utilities.io_operations import create_output_structure
from worsecrossbars.utilities.io_operations import user_folders
from worsecrossbars.utilities.logging_module import Logging
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier
from worsecrossbars.utilities.dropbox_upload import DropboxUpload


def stop_handler(signum, _):
    """
    This function handles stop signals transmitted by the Kernel when the script terminates
    abruptly/unexpectedly.
    """

    if command_line_args.log:
        log.write("Simulation terminated unexpectedly. Got signal " +
            f"{signal.Signals(signum).name}.\nEnding.\n")
        log.write(special="abruptend")

    if command_line_args.teams:
        teams.send_message(f"Using parameters:\n{simulation_parameters}",
                           title="Simulation terminated unexpectedly", color="b90e0a")

    gc.collect()
    sys.exit(0)


def main():
    """
    Main point of entry for the computing-side of the package.
    """

    # Defining percentages of faulty devices that will be simulated
    percentages = np.arange(0, 1.01, 0.01)

    if command_line_args.dropbox:
        dbx = DropboxUpload(output_folder)

    mnist_dataset = create_datasets(training_validation_ratio=3)
    weights_list, histories_list = train_models(mnist_dataset, simulation_parameters,
                                                epochs=10, batch_size=100, log=log)

    # Computing training and validation loss and accuracy by averaging over all the models trained
    # in the previous step
    training_validation_data = training_validation_metrics(histories_list)

    if command_line_args.log:
        log.write(string="Done training. Computing loss and accuracy.")

    # Saving training/validation data to file
    with open(str(Path.home().joinpath("worsecrossbars", "outputs", output_folder,
    "training_validation", f"training_validation_{fault_type}_{number_hidden_layers}HL" +
    f"_{noise_variance}NV.pickle")),"wb") as file:
        pickle.dump((training_validation_data, fault_type, number_hidden_layers, noise_variance),
                     file)

    if command_line_args.log:
        log.write(string="Saved training and validation data.")

    # Running a variety of simulations to average out stochastic variance
    accuracies = run_simulation(weights_list, percentages, mnist_dataset,
                                simulation_parameters, log)

    # Saving accuracies array to file
    with open(str(Path.home().joinpath("worsecrossbars", "outputs", output_folder, "accuracies",
    f"accuracies_{fault_type}_{number_hidden_layers}HL_{noise_variance}NV.pickle")),"wb") as file:
        pickle.dump((percentages, accuracies, fault_type, number_hidden_layers, noise_variance),
                     file)

    if command_line_args.log:
        log.write(special="end")

    if command_line_args.teams:
        teams.send_message(f"Using parameters:\n{simulation_parameters}",
                           title="Finished simulation", color="1625f3")

    if command_line_args.dropbox:
        dbx.upload()
    sys.exit(0)


if __name__ == "__main__":

    # Command line parser for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", dest="setup", metavar="INITIAL_SETUP",
        help="Run the inital setup", type=bool, default=False)
    parser.add_argument("--config", dest="config", metavar="CONFIG_FILE",
        help="Provide the config file needed for simulations", type=str)
    parser.add_argument("-w", dest="wipe_current", metavar="WIPE_CURRENT",
        help="Wipe the current output (or config)", type=bool, default=False)
    parser.add_argument("-l", dest="log", metavar="LOG",
        help="Enable logging the output in a separate file", type=bool, default=True)
    parser.add_argument("-d", dest="dropbox", metavar="DROPBOX",
        help="Enable Dropbox integration", type=bool, default=True)
    parser.add_argument("-t", dest="teams", metavar="MSTEAMS",
        help="Enable MS Teams integration", type=bool, default=True)

    command_line_args = parser.parse_args()

    if command_line_args.setup:

        main_setup(command_line_args.wipe_current)
        sys.exit(0)

    else:

        # Get the JSON supplied, parse it, validate it against a known schema.
        json_path = Path.cwd().joinpath(command_line_args.config)
        simulation_parameters = read_external_json(str(json_path))
        validate_json(simulation_parameters)
        validate_parameters(simulation_parameters)

        # Create user and output folders.
        user_folders()
        output_folder = create_output_structure(simulation_parameters,
                                                command_line_args.wipe_current)

        # Extract useful info from JSON Object
        number_hidden_layers = simulation_parameters["number_hidden_layers"]
        fault_type = simulation_parameters["fault_type"]
        noise_variance = simulation_parameters["noise_variance"]

        if command_line_args.log:
            log = Logging(simulation_parameters=simulation_parameters, output_folder=output_folder)
            log.write(special="begin")
        else:
            log = Logging()

        if command_line_args.teams:
            teams = MSTeamsNotifier(read_webhook())
            teams.send_message(f"Using parameters:\n{simulation_parameters}",
                               title="Started simulation", color="028a0f")

        # Attach Signal Handler
        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        if platform.system() == "Darwin" or platform.system() == "Linux":
            signal.signal(signal.SIGHUP, stop_handler)

        # GoTo main
        main()
