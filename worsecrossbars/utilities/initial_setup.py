import os
import argparse
from worsecrossbars import configs
import wget
from urllib.error import URLError

from worsecrossbars.utilities import auth_dropbox, msteams_notifier
from worsecrossbars.utilities import create_folder_structure

def main (overwrite_configs=False):
    """
    
    """
    
    create_folder_structure.user_folders()
    
    try:
        font = wget.download("https://github.com/nicogig/ComputerModern/raw/main/cmunrm.ttf", out=str(configs.working_dir.joinpath("utils")))
    except URLError as err:
        print(f"The Computer Modern font could not be downloaded because wget terminated unexpectedly with error {err.reason}.\nPlotting will use the standard Matplotlib font. Please consult the Docs for further guidance.")

    if overwrite_configs:
        auth_dropbox.authenticate()
        msteams_notifier.require_webhook()
    else:
        if not os.path.exists(configs.working_dir.joinpath("config", "user_secrets.json")):
            auth_dropbox.authenticate()
        if not os.path.exists(configs.working_dir.joinpath("config", "msteams.json")):
            msteams_notifier.require_webhook()

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("-ow", dest="overwrite", metavar="OVERWRITE", help="Overwrite the config files", type=bool, default=False)
    args=parser.parse_args()
    main(args.overwrite)
