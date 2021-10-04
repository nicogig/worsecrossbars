import pickle
import matplotlib.font_manager as fm
import numpy as np

from worsecrossbars import configs
from worsecrossbars.plotting.plotting import training_validation_plotter

# Importing LaTeX font for plots
try:
    font = fm.FontProperties(fname=configs.working_dir.joinpath("utils", "cmunrm.ttf"), size=18)
except FileNotFoundError:
    font = fm.FontProperties(family="sans-serif", size=18)

# Opening training/validation arrays from files
accuracy_values = []
validation_accuracy_values = []
loss_values = []
validation_loss_values = []

for number_hidden_layers in range(1, 5):

    for fault_type in range (1, 4):
        
        temp_acc = []
        temp_val_acc = []
        temp_loss = []
        temp_val_loss = []

        try:
            directory = str(configs.working_dir.joinpath("outputs", "training_validation", f"training_validation_faultType{fault_type}_{number_hidden_layers}HL.pickle"))
            temp_acc.append(pickle.load(open(directory, "rb"))[0])
            temp_val_acc.append(pickle.load(open(directory, "rb"))[1])
            temp_loss.append(pickle.load(open(directory, "rb"))[2])
            temp_val_loss.append(pickle.load(open(directory, "rb"))[3])
        except FileNotFoundError:
            print(f"Fault {number_hidden_layers}HL_{fault_type}FT not yet implemented.")
        
    accuracy_values.append(np.mean(np.array(temp_acc), axis=0))
    validation_accuracy_values.append(np.mean(np.array(temp_val_acc), axis=0))
    loss_values.append(np.mean(np.array(temp_loss), axis=0))
    validation_loss_values.append(np.mean(np.array(temp_val_loss), axis=0))

epochs = np.arange(1, len(accuracy_values[0])+1)

# Plotting training/validation curves for 1-layer topology
training_validation_plotter(epochs, accuracy_values[0], validation_accuracy_values[0], value_type="Accuracy", number_hidden_layers=1, fpath=font, save=True)
training_validation_plotter(epochs, loss_values[0], validation_loss_values[0], value_type="Loss", number_hidden_layers=1, fpath=font, save=True)

# Plotting training/validation curves for 2-layer topology
training_validation_plotter(epochs, accuracy_values[1], validation_accuracy_values[1], value_type="Accuracy", number_hidden_layers=2, fpath=font, save=True)
training_validation_plotter(epochs, loss_values[1], validation_loss_values[1], value_type="Loss", number_hidden_layers=2, fpath=font, save=True)

# Plotting training/validation curves for 3-layer topology
training_validation_plotter(epochs, accuracy_values[2], validation_accuracy_values[2], value_type="Accuracy", number_hidden_layers=3, fpath=font, save=True)
training_validation_plotter(epochs, loss_values[2], validation_loss_values[2], value_type="Loss", number_hidden_layers=3, fpath=font, save=True)

# Plotting training/validation curves for 4-layer topology
training_validation_plotter(epochs, accuracy_values[3], validation_accuracy_values[3], value_type="Accuracy", number_hidden_layers=4, fpath=font, save=True)
training_validation_plotter(epochs, loss_values[3], validation_loss_values[3], value_type="Loss", number_hidden_layers=4, fpath=font, save=True)
