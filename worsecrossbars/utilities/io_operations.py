import sys, os, json, shutil
from pathlib import Path
from datetime import datetime

def read_external_JSON (file_path):
    """
    read_external_JSON(file_name):
    Given a file_path, read the contents of the file and dump it in an object.
    """
    if not os.path.exists(file_path):
        print("The provided file does not exist! Exiting...")
        sys.exit(0)
    else:
        with open(file_path, "r", encoding="utf8") as json_file:
            json_object = json.load(json_file)
        return json_object


def user_folders():
    """
    user_folders():
    Creates the basic folder structure in the user's HOME folder.
    """
    home_dir = Path.home()
    home_dir.joinpath("worsecrossbars", "config").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "utils").mkdir(parents=True, exist_ok=True)


def create_output_structure(args):
    """
    create_output_structure(args):
    Creates an output folder given the args of the simulation.
    """
    home_dir = Path.home()
    time_now = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    folder = f"{args.number_hidden_layers}HL_{args.fault_type}FT_{args.noise}N_{args.noise_variance}NV-{time_now.__str__()}"
    if args.wipe_current:
        directory = str(home_dir.joinpath("worsecrossbars", "outputs"))
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
    home_dir.joinpath("worsecrossbars", "outputs", folder, "accuracies").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", folder, "logs").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", folder, "plots", "accuracies").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", folder, "plots", "training_validation").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", folder, "training_validation").mkdir(parents=True, exist_ok=True)
    return folder

def store_webhook ():
    """
    store_webhook():
    Asks the user for a MSTeam Webhook URL and store it.
    """
    working_dir = Path.home().joinpath("worsecrossbars")
    webhook_url = input("Please enter the MS Teams Webhook URL: ")
    jsonified_webhook = {"msteams_webhook":webhook_url}
    with open(str(working_dir.joinpath("config", "msteams.json")), 'w') as outfile:
        json.dump(jsonified_webhook, outfile)

def read_webhook ():
    """
    read_webhook():
    Reads the Webhook URL from the user's HOME directory.
    """
    working_file = Path.home().joinpath("worsecrossbars", "config", "msteams.json")
    if not os.path.exists(working_file):
        print("Please run this module with --setup before using Internet options!")
        sys.exit(0)
    else:
        with open(str(working_file)) as json_file:
            webhook_url = json.load(json_file)["msteams_webhook"]
        return webhook_url

def store_dropbox_keys ():
    """
    store_dropbox_keys():
    Asks the user for Dropbox keys, and stores them.
    """
    working_file = Path.home().joinpath("worsecrossbars", "config", "app_keys.json")
    app_key = input("Enter the APP KEY from your Dropbox App: ").strip()
    app_secret = input("Enter the APP SECRET from your Dropbox App: ").strip()
    app_keys = {"APP_KEY":app_key, "APP_SECRET":app_secret}

    with open(str(working_file), 'w', encoding="utf8") as outfile:
        json.dump(app_keys, outfile)