import numpy as np
import matplotlib.pyplot as plt
from worsecrossbars import config

def training_validation_plotter(epochs, training, validation, value_type="", number_hidden_layers=None, fpath=None, save=False, label_step=1):
    """
    training_validation_plotter:
    Plot the Training and Validation curves with respect to the Epochs.
    Inputs:
        -   epochs: The range of the epochs used to train the network.
        -   training: An array (list) containing the values obtained from the training stage.
        -   validation: An array (list) containing the values obtained from the validation stage.
        -   value_type: A string describing the type of data. Allowed values are "Accuracy" and "Loss".
        -   number_hidden_layers: An integer indicating which network topolgy is being examined.
        -   fpath: The FontProperties object containing information about the font to use.
        -   save: Boolean specifying whether the plot is to be saved to a file.
        -   label_step: The step for the labels on the x-axis. Defaults to 1.
    Output:
        -   A graph, both inline (if using Jupyter), and as a png (if the "save" flag is set to true).
    """

    if (value_type.lower() != "accuracy" and value_type.lower() != "loss"):
        raise ValueError("\"value_type\" parameter must be either \"Accuracy\" or \"Loss\".")
    
    if number_hidden_layers == None:
        raise ValueError("\"number_hidden_layers\" parameter must be passed to the plotting function.")
    
    if fpath == None:
        raise ValueError("\"fpath\" parameter must be passed to the plotting function.")

    fig = plt.figure()
    fig.set_size_inches(10, 6)

    title = f"Training and validation {value_type.lower()}, {number_hidden_layers}HL"

    if value_type.lower() == "accuracy":
        y_label = f"{value_type.title()} (%)"
        training_legend_label = f"Training {value_type.lower()} (%)"
        validation_legend_label = f"Validation {value_type.lower()} (%)"
        plt.plot(epochs, training*100, "-bo", markersize=7, label=training_legend_label, linewidth=2)
        plt.plot(epochs, validation*100, "-rD", markersize=7, label=validation_legend_label, linewidth=2)
        L = plt.legend(fontsize=15, loc="lower right")
    else:
        y_label = value_type.title()
        training_legend_label = f"Training {value_type.lower()}"
        validation_legend_label = f"Validation {value_type.lower()}"
        plt.plot(epochs, training, "-bo", markersize=7, label=training_legend_label, linewidth=2)
        plt.plot(epochs, validation, "-rD", markersize=7, label=validation_legend_label, linewidth=2)
        L = plt.legend(fontsize=15, loc="upper right")

    plt.xlabel("Epochs", font=fpath, fontsize=20)
    plt.ylabel(y_label, font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.xticks(np.arange(0, len(epochs) + 1, step=label_step), font=fpath, fontsize=15)
    plt.yticks(font=fpath, fontsize=15)
    
    plt.setp(L.texts, font=fpath)

    if save:
        plt.savefig(str(config.working_dir.parent.joinpath("outputs", "plots", "training_validation", f"training_validation_{value_type.lower()}_plot_{number_hidden_layers}HL.png")))
    
    plt.title(title, font=fpath, fontsize=20)
    plt.show()



def accuracy_curves_plotter(percentages, accuracies_list, fault_type=1, fpath=None, save=False, labels=[], label_step=10):
    """
    accuracy_curves_plotter:
    Plot the accuracy curves of different Neural Networks.
    Inputs:
        -   percentages: The x-axis datapoints.
        -   accuracies_list: A list of arrays. Each array contains accuracy values for each % of faulty devices.
        -   fault_type: An integer determining the type of fault. Acceptable values are 1, 2, or 3. Default: 1.
        -   fpath: The FontProperties object containing information about the font to use. Default: None.
        -   save: Boolean specifying whether the plot is to be saved to a file.
        -   labels: An array of strings containing the names of the curves to be plotted. Default: [].
        -   label_step: The step for the labels on the x-axis. Defaults to 10.
    Outputs:
        -   A graph, both inline (if using Jupyter), and as a png (if the "save" flag is set to true).
    """

    if fault_type not in (1, 2, 3):
        raise ValueError("\"fault_type\" parameter must be an integer comprised between 1 and 3.")
    
    if fpath == None:
        raise ValueError("\"fpath\" parameter must be passed to the plotting function.")
    
    if len(labels) != len(accuracies_list):
        raise ValueError("Not enough labels were passed to the function.")
    
    faults_tuple = ("Cannot electroform", "HRS", "LRS")

    fig = plt.figure()
    fig.set_size_inches(10, 6)

    title = f"Accuracy curves: \"{faults_tuple[fault_type-1]}\" fault"

    if fault_type == 1:
        x_label = "Percentage of devices which cannot electroform (%)"
    else:
        x_label = f"Percentage of devices which are stuck at {faults_tuple[fault_type-1]} (%)"

    for count, accuracy in enumerate(accuracies_list):
        plt.plot(percentages*100, accuracy*100, label=labels[count], linewidth=2)

    plt.xlabel(x_label, font=fpath, fontsize=20)
    plt.ylabel("Mean accuracy (%)", font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.xticks(np.arange(0, 101, step=label_step), font=fpath, fontsize=15)
    plt.yticks(np.arange(0, 101, step=label_step), font=fpath, fontsize=15)

    L = plt.legend(fontsize=15)
    
    plt.setp(L.texts, font=fpath)

    if save:
        plt.savefig(str(config.working_dir.parent.joinpath("outputs", "plots", "accuracies", f"accuracies_plot_faultType{fault_type}.png")))
    
    plt.title(title, font=fpath, fontsize=20)
    plt.show()
    