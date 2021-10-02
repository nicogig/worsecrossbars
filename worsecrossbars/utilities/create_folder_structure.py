from pathlib import Path

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