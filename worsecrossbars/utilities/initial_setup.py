"""
initial_setup.py
Module used to configure the system to use worsecrossbars.
"""

import os 
import wget
from urllib.error import URLError
from worsecrossbars import configs
from worsecrossbars.utilities import io_operations, auth_dropbox

def main_setup (overwrite_configs=False):
    """
    main_setup:
    A function to download the correct fonts, and prep the system to use worsecrossbars.
    """
    
    io_operations.user_folders()
    
    try:
        wget.download("https://github.com/nicogig/ComputerModern/raw/main/cmunrm.ttf", out=str(configs.working_dir.joinpath("utils")))
    except URLError as err:
        print(f"The Computer Modern font could not be downloaded because wget" + \
            f" terminated unexpectedly with error {err.reason}.\nPlotting will" + \
            " use the standard Matplotlib font. Please consult the Docs for further guidance.")

    if overwrite_configs:
        auth_dropbox.authenticate()
        io_operations.read_webhook()
    else:
        if not os.path.exists(configs.working_dir.joinpath("config", "user_secrets.json")):
            auth_dropbox.authenticate()
        if not os.path.exists(configs.working_dir.joinpath("config", "msteams.json")):
            io_operations.read_webhook()
