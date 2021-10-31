"""
MLP.py (Temporary Name)
Worsecrossbars main module and entrypoint.
"""

import argparse
import sys
from pathlib import Path
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from worsecrossbars.utilities import initial_setup
from worsecrossbars.utilities import io_operations

def main():
    """
    main(command_line_args):
    The main function.
    """
    json_schema = {
        "type": "object",
        "properties" : {
            "HRS_LRS_ratio": {"type" : "integer"},
            "number_of_conductance_levels": {"type": "integer"},
            "excluded_weights_proportion" : {"type": "number"},
            "number_hidden_layers": {"type": "integer"},
            "fault_type": {"type": "string"},
            "noise_variance": {"type": "number"},
            "number_ANNs": {"type": "integer"},
            "number_simulations": {"type": "integer"}
        }
    }
    json_path = Path.cwd().joinpath(command_line_args.config)
    extracted_json = io_operations.read_external_json(str(json_path))
    try:
        validate(extracted_json, json_schema)
    except ValidationError as err:
        print(f"{err.message}")
        sys.exit(0)
    pass

if __name__ == "__main__":

    # Command line parser for input arguments
    parser=argparse.ArgumentParser()

    parser.add_argument("--setup", dest="setup", metavar="INITIAL_SETUP", \
         help="Run the inital setup", type=bool, default=False)
    parser.add_argument("--config", dest="config", metavar="CONFIG_FILE", \
         help="Provide the config file needed for simulations", type=str)
    parser.add_argument("-w", dest="wipe_current", metavar="WIPE_CURRENT", \
         help="Wipe the current configuration", type=bool, default=False)
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
        main()
