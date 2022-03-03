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

import tensorflow as tf
from numpy import ndarray

from worsecrossbars.backend.mlp_trainer import mnist_datasets
from worsecrossbars.backend.simulation import run_simulations
from worsecrossbars.utilities.dropbox_upload import DropboxUpload
from worsecrossbars.utilities.initial_setup import main_setup
from worsecrossbars.utilities.io_operations import create_output_structure
from worsecrossbars.utilities.io_operations import read_external_json
from worsecrossbars.utilities.io_operations import read_webhook
from worsecrossbars.utilities.io_operations import user_folders
from worsecrossbars.utilities.json_handlers import validate_json
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier
from worsecrossbars.utilities import nvidia


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
    tf_device: str = "cpu:0",
    _teams: MSTeamsNotifier = None,
    _batch_size: int = 100,
):
    """A worker, an async class that handles the heavy-lifting computation-wise."""

    process_id = os.getpid()

    logging.info("Attempting simulation with process ID %d.", process_id)

    if _teams:
        _teams.send_message(
            f"Process ID: {process_id}\nSimulation parameters:\n{simulation_parameters}",
            title="Started simulation",
            color="ffca33",
        )

    # Running simulations
    with tf.device(tf_device):
        accuracies, pre_discretisation_accuracies = run_simulations(simulation_parameters, dataset, batch_size=_batch_size)

    # Saving accuracies array to file
    with open(
        str(
            Path.home().joinpath(
                "worsecrossbars",
                "outputs",
                _output_folder,
                f"output_{process_id}.json",
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

    if _teams:
        _teams.send_message(
            f"Process ID: {process_id}",
            title="Finished simulation",
            color="1fd513",
        )


def main():
    """Main point of entry for the computing-side of the package."""

    if command_line_args.dropbox:
        dbx = DropboxUpload(output_folder)

    dataset = mnist_datasets(training_validation_ratio=3)

    if tf.config.list_physical_devices("GPU"):
        # Perform a different parallelisation strategy if on GPU
        # -nicogig
        pool = []

        for simulation_parameters in json_object["simulations"]:
            
            next_available_gpu = nvidia.pick_gpu_lowest_memory()
            tf_gpu = "gpu:" + str(next_available_gpu)
            
            if command_line_args.teams is None:
                process = Process(
                    target=worker, args=[dataset, simulation_parameters, output_folder, tf_gpu]
                )
            else:
                process = Process(
                    target=worker, args=[dataset, simulation_parameters, output_folder, tf_gpu, teams]
                )
            process.start()
            pool.append(process)

        for process in pool:
            process.join()
    else:

        pool = []

        for simulation_parameters in json_object["simulations"]:
            if command_line_args.teams is None:
                process = Process(
                    target=worker, args=[dataset, simulation_parameters, output_folder, "cpu:0"]
                )
            else:
                process = Process(
                    target=worker, args=[dataset, simulation_parameters, output_folder, "cpu:0", teams]
                )
            process.start()
            pool.append(process)

        for process in pool:
            process.join()

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
