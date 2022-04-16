"""compute:
Worsecrossbars' main module and entrypoint.
"""
import json
import os
import platform
import sys
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from numpy import ndarray

from worsecrossbars.backend.mlp_trainer import get_dataset
from worsecrossbars.backend.simulation import run_simulations
from worsecrossbars.utilities.dropbox_upload import DropboxUpload
from worsecrossbars.utilities.logging_module import Logging
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier


# Unified Worker
def worker(
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    simulation_parameters: dict,
    _output_folder: str,
    horovod: bool,
    _teams: MSTeamsNotifier = None,
    _logger: Logging = None,
    _batch_size: int = 100,
):
    """A worker, an async class that handles the heavy-lifting computation-wise."""

    process_id = os.getpid()

    if horovod:
        # pylint: disable=import-outside-toplevel
        import horovod.tensorflow as hvd

        # pylint: enable=import-outside-toplevel

    if not horovod or (horovod and hvd.rank() == 0):
        if _logger:
            _logger.write(f"Attempting simulation with process ID {process_id}")

        if _teams:
            _teams.send_message(
                f"Process ID: {process_id}\nSimulation parameters:\n{simulation_parameters}",
                title="Started simulation",
                color="ffca33",
            )

    accuracies, pre_discretisation_accuracies = run_simulations(
        simulation_parameters, dataset, batch_size=_batch_size, horovod=horovod, logger=_logger
    )

    if not horovod or (horovod and hvd.rank() == 0):
        with open(
            str(
                Path.home().joinpath(
                    "worsecrossbars",
                    "outputs",
                    _output_folder,
                    "accuracies",
                    f"output_{process_id}_{simulation_parameters['ID']}.json",
                )
            ),
            "w",
            encoding="utf-8",
        ) as file:
            if simulation_parameters["discretisation"]:
                output_object = {
                    "pre_discretisation_accuracies": pre_discretisation_accuracies.tolist(),
                    "accuracies": accuracies.tolist(),
                    "simulation_parameters": simulation_parameters,
                }
            else:
                output_object = {
                    "accuracies": accuracies.tolist(),
                    "simulation_parameters": simulation_parameters,
                }
            json.dump(output_object, file)

        if _logger:
            _logger.write(f"Saved accuracy data for simulation with process ID {process_id}.")

        if _teams:
            _teams.send_message(
                f"Process ID: {process_id}",
                title="Finished simulation",
                color="1fd513",
            )


# Main
def main(command_line_args, output_folder, json_object, horovod, **kwargs):
    """Main point of entry for the computing-side of the package."""

    teams = kwargs.get("teams", None)
    logger = kwargs.get("logger", None)

    if horovod:
        # pylint: disable=import-outside-toplevel
        import horovod.tensorflow as hvd

        # pylint: enable=import-outside-toplevel

        hvd.init()

    if command_line_args.dropbox:
        dbx = DropboxUpload(output_folder)

    dataset = get_dataset("mnist", 3)

    if horovod:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    for index, simulation_parameters in enumerate(json_object["simulations"]):

        simulation_parameters["ID"] = index + 1

        if horovod:
            worker(dataset, simulation_parameters, output_folder, horovod, teams, logger)
        else:
            if platform.system() == "Darwin":
                dev = "/device:cpu:0"
            else:
                dev = "/device:gpu:1"

            with tf.device(dev):
                worker(dataset, simulation_parameters, output_folder, horovod, teams, logger)

    if command_line_args.dropbox:
        dbx.upload()
        logger.write("Uploaded simulation outcome to Dropbox.")
        if command_line_args.teams:
            teams.send_message(
                f"Simulations {output_folder} uploaded successfully.",
                title="Uploaded to Dropbox",
                color="0060ff",
            )

    sys.exit(0)
