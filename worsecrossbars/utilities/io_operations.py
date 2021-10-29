import sys, os, json, shutil
from worsecrossbars import configs
from pathlib import Path
from datetime import datetime

def user_folders():
    """
    
    """
    
    home_dir = Path.home()
    home_dir.joinpath("worsecrossbars", "config").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "utils").mkdir(parents=True, exist_ok=True)


def create_output_structure(args):
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

    """
    webhook_url = input("Please enter the MS Teams Webhook URL: ")
    jsonified_webhook = {"msteams_webhook":webhook_url}
    with open(str(configs.working_dir.joinpath("config", "msteams.json")), 'w') as outfile:
        json.dump(jsonified_webhook, outfile)

def read_webhook ():
    if not os.path.exists(configs.working_dir.joinpath("config", "msteams.json")):
        print("Please run this module with --setup before using Internet options!")
        sys.exit(0)
    else:
        with open(str(configs.working_dir.joinpath("config", "msteams.json"))) as json_file:
            webhook_url = json.load(json_file)["msteams_webhook"]
        return webhook_url

def store_dropbox_keys ():
    app_key = input("Enter the APP KEY from your Dropbox App: ").strip()
    app_secret = input("Enter the APP SECRET from your Dropbox App: ").strip()
    app_keys = {"APP_KEY":app_key, "APP_SECRET":app_secret}

    with open(str(configs.working_dir.joinpath("config", "app_keys.json")), 'w') as outfile:
        json.dump(app_keys, outfile)