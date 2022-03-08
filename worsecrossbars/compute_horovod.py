"""compute:
Worsecrossbars' main module and entrypoint.
"""
import argparse
import gc
import json
import logging
import os
import platform
import signal
import sys
from multiprocessing import Process
from pathlib import Path
from typing import Tuple

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from numpy import ndarray

from worsecrossbars.backend.mlp_trainer import mnist_datasets
from worsecrossbars.backend.nonidealities import D2DVariability
from worsecrossbars.backend.nonidealities import IVNonlinear
from worsecrossbars.backend.nonidealities import StuckAtValue
from worsecrossbars.backend.nonidealities import StuckDistribution
from worsecrossbars.backend.simulation import run_simulations
from worsecrossbars.utilities.dropbox_upload import DropboxUpload
from worsecrossbars.utilities.initial_setup import main_setup
from worsecrossbars.utilities.io_operations import create_output_structure
from worsecrossbars.utilities.io_operations import read_external_json
from worsecrossbars.utilities.io_operations import read_webhook
from worsecrossbars.utilities.io_operations import user_folders
from worsecrossbars.utilities.json_handlers import validate_json
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier


def gen_nonideality_list(simulation_parameters):
    nonidealities = []

    while simulation_parameters["nonidealities"] > 0:
        nonideality = simulation_parameters["nonidealities"].pop()

        if nonideality["type"] == "IVNonlinear":
            nonidealities.append(
                IVNonlinear(
                    V_ref=nonideality["parameters"][0],
                    avg_gamma=nonideality["parameters"][1],
                    std_gamma=nonideality["parameters"][2],
                )
            )
        elif nonideality["type"] == "D2DVariability":
            nonidealities.append(
                D2DVariability(
                    simulation_parameters["G_off"],
                    simulation_parameters["G_on"],
                    nonideality["parameters"][0],
                    nonideality["parameters"][1],
                )
            )
        elif nonideality["type"] == "StuckAtValue":
            nonidealities.append(StuckAtValue(value=nonideality["parameters"][0]))
            simulation_parameters["nonidealities"].remove(nonideality)
        else:
            if isinstance(nonideality["parameters"][0], list):
                nonidealities.append(StuckDistribution(distrib=nonideality["parameters"][0]))
                simulation_parameters["nonidealities"].remove(nonideality)
            elif isinstance(nonideality["parameters"][0], int):
                nonidealities.append(
                    StuckDistribution(
                        num_of_weights=nonideality["parameters"][0],
                        G_off=simulation_parameters["G_off"],
                        G_on=simulation_parameters["G_on"],
                    )
                )
    return nonidealities


def stop_handler(signum, _):
    """This function handles stop signals transmitted by the Kernel when the script terminates
    abruptly/unexpectedly."""

    logging.error(
        "Simulation terminated unexpectedly due to Signal %s",
        signal.Signals(signum).name,
    )
    if command_line_args.teams:
        sims = json_object["simulations"]
        teams.send_message(
            f"Using parameters:\n{sims}\nSignal:{signal.Signals(signum).name}",
            title="Simulation terminated unexpectedly",
            color="b90e0a",
        )
    gc.collect()
    sys.exit(1)


def worker(
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    simulation_parameters: dict,
    _output_folder: str,
    horovod: bool = False,
    _teams: MSTeamsNotifier = None,
    _batch_size: int = 100,
):
    """A worker, an async class that handles the heavy-lifting computation-wise."""

    process_id = os.getpid()

    logging.info("Attempting simulation with process ID %d.", process_id)

    if _teams and hvd.rank() == 0:
        _teams.send_message(
            f"Process ID: {process_id}\nSimulation parameters:\n{simulation_parameters}",
            title="Started simulation",
            color="ffca33",
        )

    # Running simulations
    accuracies, pre_discretisation_accuracies = run_simulations(
        simulation_parameters, dataset, batch_size=_batch_size, horovod=horovod
    )

    if horovod and hvd.rank() == 0:
        # Saving accuracies array to file
        with open(
            str(
                Path.home().joinpath(
                    "worsecrossbars",
                    "outputs",
                    _output_folder,
                    f"output_{process_id}_{simulation_parameters['number_hidden_layers']}.json",
                )
            ),
            "w",
            encoding="utf-8",
        ) as file:
            output_object = {
                "pre_discretisation_accuracies": pre_discretisation_accuracies.tolist(),
                "accuracies": accuracies.tolist(),
                "simulation_parameters": simulation_parameters,
            }
            json.dump(output_object, file)

        logging.info("Saved accuracy data for simulation with process ID %d.", process_id)

    if _teams and hvd.rank() == 0:
        _teams.send_message(
            f"Process ID: {process_id}",
            title="Finished simulation",
            color="1fd513",
        )


def main():
    """Main point of entry for the computing-side of the package."""
    # tf.debugging.set_log_device_placement(True)
    hvd.init()

    if command_line_args.dropbox:
        dbx = DropboxUpload(output_folder)

    dataset = mnist_datasets(training_validation_ratio=3)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.set_visible_devices(gpus[hvd.local_rank() + 1], "GPU")

    for simulation_parameters in json_object["simulations"]:
        worker(dataset, simulation_parameters, output_folder, True, teams)

    if command_line_args.dropbox:
        dbx.upload()
        logging.info("Uploaded simulation outcome to Dropbox.")
        if command_line_args.teams:
            teams.send_message(
                f"Simulations {output_folder} uploaded successfully.",
                title="Uploaded to Dropbox",
                color="0060ff",
            )
    sys.exit(0)


if __name__ == "__main__":

    # Command line parser for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        metavar="CONFIG_FILE",
        nargs="?",
        help="Provide the config file needed for simulations",
        type=str,
        default="",
    )
    parser.add_argument(
        "--setup",
        dest="setup",
        metavar="INITIAL_SETUP",
        help="Run the inital setup",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-w",
        dest="wipe_current",
        metavar="WIPE_CURRENT",
        help="Wipe the current output (or config)",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-d",
        dest="dropbox",
        metavar="DROPBOX",
        help="Enable Dropbox integration",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-t",
        dest="teams",
        metavar="MSTEAMS",
        help="Enable MS Teams integration",
        type=bool,
        default=True,
    )

    command_line_args = parser.parse_args()

    if command_line_args.setup:

        main_setup(command_line_args.wipe_current)
        sys.exit(0)

    else:

        # Create user and output folders.
        user_folders()
        output_folder = create_output_structure(command_line_args.wipe_current)

        logging.basicConfig(
            filename=str(
                Path.home().joinpath("worsecrossbars", "outputs", output_folder, "logs", "run.log")
            ),
            filemode="w",
            format="[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s",
            level=logging.INFO,
            datefmt="%d-%b-%y %H:%M:%S",
        )

        # Get the JSON supplied, parse it, validate it against a known schema.
        json_path = Path.cwd().joinpath(command_line_args.config)
        json_object = read_external_json(str(json_path))
        validate_json(json_object)

        if command_line_args.teams:
            teams = MSTeamsNotifier(read_webhook())

        # Attach Signal Handler
        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        if platform.system() == "Darwin" or platform.system() == "Linux":
            signal.signal(signal.SIGHUP, stop_handler)

        # GoTo main
        main()
