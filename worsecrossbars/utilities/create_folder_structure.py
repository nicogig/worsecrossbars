from pathlib import Path
import os, shutil

def user_folders():
    """
    
    """
    
    home_dir = Path.home()
    home_dir.joinpath("worsecrossbars", "outputs", "accuracies").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "logs").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "plots", "accuracies").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "plots", "training_validation").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "training_validation").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "config").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "utils").mkdir(parents=True, exist_ok=True)


def remove_existing_files():
    """
    
    """

    home_dir = Path.home()
    directory = str(home_dir.joinpath("worsecrossbars", "outputs"))
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    home_dir.joinpath("worsecrossbars", "outputs", "accuracies").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "logs").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "plots", "accuracies").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "plots", "training_validation").mkdir(parents=True, exist_ok=True)
    home_dir.joinpath("worsecrossbars", "outputs", "training_validation").mkdir(parents=True, exist_ok=True)
    
