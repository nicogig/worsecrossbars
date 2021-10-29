from pathlib import Path
import os, shutil
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
    
