from pathlib import Path

def user_folders():
    """
    
    """
    
    home_dir = Path.home()
    home_dir.mkdir("worsecrossbars", parents=True, exist_ok=True)
    
    py_folder = home_dir.joinpath("worsecrossbars")
    py_folder.mkdir("outputs", parents=True, exist_ok=True)
    py_folder.mkdir("config", parents=True, exist_ok=True)

    outputs_folder = py_folder.joinpath("outputs")
    outputs_folder.mkdir("accuracies", parents=True, exist_ok=True)
    outputs_folder.mkdir("logs", parents=True, exist_ok=True)
    outputs_folder.mkdir("plots", parents=True, exist_ok=True)
    outputs_folder.mkdir("training_validation", parents=True, exist_ok=True)
    
    plots_folder = outputs_folder.joinpath("plots")
    plots_folder.mkdir("accuracies", parents=True, exist_ok=True)
    plots_folder.mkdir("training_validation", parents=True, exist_ok=True)