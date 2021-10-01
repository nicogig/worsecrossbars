import os
import numpy as np
import pickle
from pathlib import Path
import matplotlib.font_manager as fm

from plotting import accuracy_curves_plotter

# Importing Latex font for plots
fpath = Path('cmunrm.ttf')
font = fm.FontProperties(fname='cmunrm.ttf', size=18)

# Opening accuracies arrays from files
data = [[], [], []]
accuracy_labels = []

for number_hidden_layers in range(1, 5):
    accuracy_labels.append("{} hidden layers".format(number_hidden_layers) if (number_hidden_layers != 1) else "1 hidden layer")

    for fault_type in range (1, 4):
        try:
            data[fault_type-1].append(pickle.load(open(f"../../outputs/accuracies/accuracies_faultType{fault_type}_{number_hidden_layers}HL.pickle", "rb")))
        except FileNotFoundError:
            print(f"Fault {number_hidden_layers}HL_{fault_type}FT not yet implemented.")

try:
    data_cannot_electroform = data[0]
    data_stuck_at_HRS = data[1]
    data_stuck_at_LRS = data[2]
except IndexError:
    print("One of the necessary files is missing.")

# Plotting accuracy curves for Type I (Cannot Electroform) fault
accuracy_curves_plotter(data_cannot_electroform[0][0], [data_cannot_electroform[i][1] for i in range(0,4)], value_type=data_cannot_electroform[0][2], fpath=font, save=True, labels=accuracy_labels)

# Plotting accuracy curves for Type II (Stuck at HRS) fault
accuracy_curves_plotter(data_stuck_at_HRS[0][0], [data_stuck_at_HRS[i][1] for i in range(0,4)], value_type=data_stuck_at_HRS[0][2], fpath=font, save=True, labels=accuracy_labels)

# Plotting accuracy curves for Type III (Stuck at LRS) fault
accuracy_curves_plotter(data_stuck_at_LRS[0][0], [data_stuck_at_LRS[i][1] for i in range(0,4)], value_type=data_stuck_at_LRS[0][2], fpath=font, save=True, labels=accuracy_labels)
