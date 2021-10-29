import os, wget
from worsecrossbars import configs
from urllib.error import URLError
from worsecrossbars.utilities import create_folder_structure, io_operations, auth_dropbox


def main_setup (overwrite_configs=False):
    """
    
    """
    
    create_folder_structure.user_folders()
    
    try:
        font = wget.download("https://github.com/nicogig/ComputerModern/raw/main/cmunrm.ttf", out=str(configs.working_dir.joinpath("utils")))
    except URLError as err:
        print(f"The Computer Modern font could not be downloaded because wget terminated unexpectedly with error {err.reason}.\nPlotting will use the standard Matplotlib font. Please consult the Docs for further guidance.")

    if overwrite_configs:
        auth_dropbox.authenticate()
        io_operations.read_webhook()
    else:
        if not os.path.exists(configs.working_dir.joinpath("config", "user_secrets.json")):
            auth_dropbox.authenticate()
        if not os.path.exists(configs.working_dir.joinpath("config", "msteams.json")):
            io_operations.read_webhook()
