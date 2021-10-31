"""
MLP.py (Temporary Name)
Worsecrossbars main module and entrypoint.
"""

import argparse
import sys
import signal
import gc
import platform
from pathlib import Path
from worsecrossbars.utilities import initial_setup, json_handlers
from worsecrossbars.utilities import io_operations
from worsecrossbars.utilities import Logging
from worsecrossbars.utilities import MSTeamsNotifier

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

def main():
    """
    main(command_line_args):
    The main function.
    """
    pass

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
        json_path = Path.cwd().joinpath(command_line_args.config)
        extracted_json = io_operations.read_external_json(str(json_path))
        json_handlers.validate_json(extracted_json)
        io_operations.user_folders()
        output_folder = io_operations.create_output_structure(extracted_json, 
        command_line_args.wipe_current)
        if command_line_args.log:
            log = Logging(extracted_json, output_folder)
            log.write(special="begin")
        if command_line_args.teams:
            teams = MSTeamsNotifier(io_operations.read_webhook())
            number_hidden_layers = extracted_json["number_hidden_layers"]
            HIDDEN_LAYER = "hidden layer" if number_hidden_layers == 1 else "hidden layers"
            fault_type = extracted_json["fault_type"]
            teams.send_message(f"Using parameters: {number_hidden_layers} {HIDDEN_LAYER}," + \
                f" fault type {fault_type}.", title="Started new simulation", color="028a0f")
        signal.signal(signal.SIGINT, stop_handler)
        signal.signal(signal.SIGTERM, stop_handler)
        if platform.system() == "Darwin" or platform.system() == "Linux":
            signal.signal(signal.SIGHUP, stop_handler)
        main()
