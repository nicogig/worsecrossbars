import pickle
import matplotlib.font_manager as fm
import os

from worsecrossbars import configs
from worsecrossbars.plotting.plotting import accuracy_curves_plotter

# Importing LaTeX font for plots
if os.path.exists(configs.working_dir.joinpath("utils", "cmunrm.ttf")):
    font = fm.FontProperties(fname=configs.working_dir.joinpath("utils", "cmunrm.ttf"), size=18)
else:
    font = fm.FontProperties(family="sans-serif", size=18)

# Opening accuracies arrays from files
data = [[[], [], []], [[], [], []]]
accuracy_labels = []

for noise_idx, noise in enumerate([True, False]):

    for number_hidden_layers in range(1, 5):

        if noise:
            accuracy_labels.append(f"{number_hidden_layers} hidden layers" if (number_hidden_layers != 1) else f"1 hidden layer")

        for fault_type in range (1, 4):
            try:
                data[noise_idx][fault_type-1].append(pickle.load(open(str(configs.working_dir.joinpath("outputs", "accuracies", f"accuracies_faultType{fault_type}_{number_hidden_layers}HL_{noise}N_1NV.pickle")), "rb")))
            except FileNotFoundError:
                print(f"File for fault {number_hidden_layers}HL_{fault_type}FT_{noise}N is not present.")


data_cannot_electroform = data[1][0]
data_cannot_electroform_noise = data[0][0]
data_stuck_at_HRS = data[1][1]
data_stuck_at_HRS_noise = data[0][1]
data_stuck_at_LRS = data[1][2]
data_stuck_at_LRS_noise = data[0][2]

try:
    # Plotting accuracy curves for regular data
    accuracy_curves_plotter(data_cannot_electroform[0][0], [data_cannot_electroform[i][1] for i in range(0,4)], fault_type=data_cannot_electroform[0][2], fpath=font, save=True, labels=accuracy_labels)
    accuracy_curves_plotter(data_stuck_at_HRS[0][0], [data_stuck_at_HRS[i][1] for i in range(0,4)], fault_type=data_stuck_at_HRS[0][2], fpath=font, save=True, labels=accuracy_labels)
    accuracy_curves_plotter(data_stuck_at_LRS[0][0], [data_stuck_at_LRS[i][1] for i in range(0,4)], fault_type=data_stuck_at_LRS[0][2], fpath=font, save=True, labels=accuracy_labels)
except IndexError:
    print("Some necessary file is missing: complete accuracy plot cannot be produced.")

try:
    # Plotting accuracy for AWGN data
    accuracy_curves_plotter(data_cannot_electroform_noise[0][0], [data_cannot_electroform_noise[i][1] for i in range(0,4)], fault_type=data_cannot_electroform_noise[0][2], noise=True, fpath=font, save=True, labels=accuracy_labels)
    accuracy_curves_plotter(data_stuck_at_HRS_noise[0][0], [data_stuck_at_HRS_noise[i][1] for i in range(0,4)], fault_type=data_stuck_at_HRS_noise[0][2], noise=True, fpath=font, save=True, labels=accuracy_labels)
    accuracy_curves_plotter(data_stuck_at_LRS_noise[0][0], [data_stuck_at_LRS_noise[i][1] for i in range(0,4)], fault_type=data_stuck_at_LRS_noise[0][2], noise=True, fpath=font, save=True, labels=accuracy_labels)
except IndexError:
    print("Some necessary file is missing: complete accuracy plot cannot be produced.")


