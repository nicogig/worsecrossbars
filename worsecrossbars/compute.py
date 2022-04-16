"""compute:
Worsecrossbars' main module and entrypoint.
"""
import argparse
import gc
import platform
import signal
import sys
from pathlib import Path
from typing import Union

from worsecrossbars.utilities.initial_setup import main_setup
from worsecrossbars.utilities.io_operations import create_output_structure
from worsecrossbars.utilities.io_operations import read_external_json
from worsecrossbars.utilities.io_operations import read_webhook
from worsecrossbars.utilities.io_operations import user_folders
from worsecrossbars.utilities.json_handlers import validate_json
from worsecrossbars.utilities.logging_module import Logging
from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier
from worsecrossbars.workers import traditional_worker


def stop_handler(signum, _):
    """This function handles stop signals transmitted by the Kernel when the script terminates
    abruptly/unexpectedly."""

    logger.write(
        f"Simulation terminated unexpectedly due to Signal {signal.Signals(signum).name}", "ERROR"
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


def main():
    """Main point of entry for the computing-side of the package."""

    if command_line_args.multiGPU:
        from worsecrossbars.workers import multi_gpu_worker

        multi_gpu_worker.main(command_line_args, output_folder, json_object, teams, logger)
    else:
        traditional_worker.main(command_line_args, output_folder, json_object, teams, logger)


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
    parser.add_argument(
        "--multiGPU",
        dest="multiGPU",
        metavar="MULTIGPU",
        help="Enable MultiGPU Support",
        type=bool,
        default=False,
    )

    command_line_args = parser.parse_args()

    if command_line_args.setup:

        main_setup(command_line_args.wipe_current)
        sys.exit(0)

    else:

        # Create user and output folders.
        user_folders()
        output_folder = create_output_structure(command_line_args.wipe_current)

        logger = Logging(output_folder)

        # Get the JSON supplied, parse it, validate it against a known schema.
        json_path = Path.cwd().joinpath(command_line_args.config)
        json_object = read_external_json(str(json_path))
        validate_json(json_object)

        if command_line_args.teams:
            teams: Union[MSTeamsNotifier, None] = MSTeamsNotifier(read_webhook())
        else:
            teams = None

        # Attach Signal Handler
        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        if platform.system() == "Darwin" or platform.system() == "Linux":
            signal.signal(signal.SIGHUP, stop_handler)

        # GoTo main
        main()
