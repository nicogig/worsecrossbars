"""
MLP.py (Temporary Name)
Worsecrossbars main module and entrypoint.
"""

import argparse
import sys
from worsecrossbars.utilities import initial_setup

if __name__ == "__main__":

    # Command line parser for input arguments
    parser=argparse.ArgumentParser()

    parser.add_argument("--setup", dest="setup", metavar="INITIAL_SETUP", help="Run the inital setup.", type=bool, default=False)
    parser.add_argument("--config", dest="config", metavar="CONFIG_FILE", help="Provide the config file needed for simulations.", type=str)
    parser.add_argument("-w", dest="wipe_current", metavar="WIPE_CURRENT", help="Wipe the current configuration.", type=bool, default=False)
    parser.add_argument("-l", dest="log", metavar="LOG", help="Enable logging the output in a separate file", type=bool, default=True)
    parser.add_argument("-d", dest="dropbox", metavar="DROPBOX", help="Enable Dropbox integration", type=bool, default=True)
    parser.add_argument("-t", dest="teams", metavar="MSTEAMS", help="Enable MS Teams integration", type=bool, default=True)

    command_line_args = parser.parse_args()

    if command_line_args.setup:
        initial_setup.main_setup(command_line_args.wipe_current)
        sys.exit(0)
    
