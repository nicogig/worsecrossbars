"""compute:
Worsecrossbars' main module and entrypoint.
"""

import json
import logging
import os
import sys
from multiprocessing import Process
from pathlib import Path
from typing import Tuple

from numpy import ndarray

from worsecrossbars.backend.mlp_trainer import mnist_datasets
from worsecrossbars.backend.simulation import run_simulations
from worsecrossbars.utilities.dropbox_upload import DropboxUpload
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier

def worker(
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    simulation_parameters: dict,
    _output_folder: str,
    _teams: MSTeamsNotifier = None,
    _logger: logging.Logger = None,
    _batch_size: int = 100,
):
    """A worker, an async class that handles the heavy-lifting computation-wise."""

    process_id = os.getpid()

    _logger.write(f"Attempting simulation with process ID {process_id}")

    if _teams:
        _teams.send_message(
            f"Process ID: {process_id}\nSimulation parameters:\n{simulation_parameters}",
            title="Started simulation",
            color="ffca33",
        )

    # Running simulations
    accuracies, pre_discretisation_accuracies = run_simulations(
        simulation_parameters, dataset, batch_size=_batch_size, logger=_logger
    )

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

    _logger.write(f"Saved accuracy data for simulation with process ID {process_id}")

    if _teams:
        _teams.send_message(
            f"Process ID: {process_id}",
            title="Finished simulation",
            color="1fd513",
        )


def main(command_line_args, output_folder, json_object, teams=None, logger=None):
    """Main point of entry for the computing-side of the package."""

    if command_line_args.dropbox:
        dbx = DropboxUpload(output_folder)

    dataset = mnist_datasets(training_validation_ratio=3)
    
    pool = []

    for simulation_parameters in json_object["simulations"]:

        worker(dataset, simulation_parameters, output_folder, teams, logger)
        
        #if command_line_args.teams is None:
        #    process = Process(
        #        target=worker, args=[dataset, simulation_parameters, output_folder, None, logger]
        #    )
        #else:
        #    process = Process(
        #        target=worker,
        #        args=[dataset, simulation_parameters, output_folder, teams, logger],
        #    )
        #process.start()
        #pool.append(process)

    #for process in pool:
    #    process.join()

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
