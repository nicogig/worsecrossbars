"""
plotting:
A plotting module used to create and save training/validation and accuracy plots.
"""

import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# def training_validation_plotter(
#     epochs,
#     training,
#     validation,
#     value_type="",
#     number_hidden_layers=None,
#     fpath=None,
#     save=False,
#     label_step=1):
#     """
#     training_validation_plotter:
#     Plot the Training and Validation curves with respect to the Epochs.
#     Inputs:
#         -   epochs: The range of the epochs used to train the network.
#         -   training: An array (list) containing the values obtained from the training stage.
#         -   validation: An array (list) containing the values obtained from the validation stage.
#         -   value_type: A string describing the type of data.
#         -   Allowed values are "Accuracy" and "Loss".
#         -   number_hidden_layers: An integer indicating which network topolgy is being examined.
#         -   fpath: The FontProperties object containing information about the font to use.
#         -   save: Boolean specifying whether the plot is to be saved to a file.
#         -   label_step: The step for the labels on the x-axis. Defaults to 1.
#     Output:
#         -   A graph, both inline (if using Jupyter)
#         -   and as a png (if the "save" flag is set to true).
#     """

#     if (value_type.lower() != "accuracy" and value_type.lower() != "loss"):
#         raise ValueError("\"value_type\" parameter must be either \"Accuracy\" or \"Loss\".")

#     if number_hidden_layers is None:
#         raise ValueError("\"number_hidden_layers\" parameter must" + \
#             " be passed to the plotting function.")

#     if fpath is None:
#         raise ValueError("\"fpath\" parameter must be passed to the plotting function.")

#     fig = plt.figure()
#     fig.set_size_inches(10, 6)

#     title = f"Training and validation {value_type.lower()}, {number_hidden_layers}HL"

#     if value_type.lower() == "accuracy":
#         y_label = f"{value_type.title()} (%)"
#         training_legend_label = f"Training {value_type.lower()} (%)"
#         validation_legend_label = f"Validation {value_type.lower()} (%)"
#         plt.plot(
#             epochs,
#             training*100,
#             "-bo",
#             markersize=7,
#             label=training_legend_label,
#             linewidth=2)
#         plt.plot(
#             epochs,
#             validation*100,
#             "-rD",
#             markersize=7,
#             label=validation_legend_label,
#             linewidth=2)
#         legend = plt.legend(fontsize=15, loc="lower right")
#     else:
#         y_label = value_type.title()
#         training_legend_label = f"Training {value_type.lower()}"
#         validation_legend_label = f"Validation {value_type.lower()}"
#         plt.plot(epochs, training, "-bo", markersize=7, label=training_legend_label, linewidth=2)
#         plt.plot(
#             epochs,
#             validation,
#             "-rD",
#             markersize=7,
#             label=validation_legend_label,
#             linewidth=2)
#         legend = plt.legend(fontsize=15, loc="upper right")

#     plt.xlabel("Epochs", font=fpath, fontsize=20)
#     plt.ylabel(y_label, font=fpath, fontsize=20)
#     plt.grid()
#     plt.tight_layout()
#     plt.xticks(np.arange(0, len(epochs) + 1, step=label_step), font=fpath, fontsize=15)
#     plt.yticks(font=fpath, fontsize=15)

#     plt.setp(legend.texts, font=fpath)

#     if save:
#         plt.savefig(
#             str(Path.home().joinpath("worsecrossbars",
#             "outputs",
#             "plots",
#             "training_validation",
#             f"training_validation_{value_type.lower()}_plot_{number_hidden_layers}HL.png")),
#             dpi=200)

#     plt.title(title, font=fpath, fontsize=20)
#     plt.show()


def accuracy_curves_plotter(accuracies_objects_list, fpath=None, x_label="", title="", filename=""):
    """
    This function plots accuracy curves with the given data.

    As a reminder, the following statements are true about the accuracies_objects contained in the
    accuracies_objects_list argument:
        percentages = accuracies_object[0]
        accuracies = accuracies_object[1]
        fault_type = accuracies_object[2]
        number_hidden_layers = accuracies_object[3]
        noise_variance = accuracies_object[4]

    Args:
      accuracies_objects_list: List containing the accuracies_objects that are to be plotted.
        Details about said objects' structure is  provided above.
      fpath: Object containing information regarding the plot's font.
      x_label: String, label used on the x axis.
      title: String, title used for the plot.
      filename: String, name used to save the plot to file. If it is not provided (or is not a
        string), the plot is not saved to file.
    """

    if fpath is None:
        warnings.warn("Remember to pass an \"fpath\" parameter to control the plot's font.")

    if not isinstance(x_label, str) or x_label == "":
        x_label = "Percentage of faulty devices (%)"

    if not isinstance(title, str) or title == "":
        title = "Influence of faulty devices on ANN inference accuracy"

    if not isinstance(filename, str):
        raise ValueError("\"filename\" parameter must be a valid string.")

    fig = plt.figure()
    fig.set_size_inches(10, 6)

    for accuracies_object in accuracies_objects_list:
        label = f"{accuracies_object[2]}, {accuracies_object[3]}HL, {accuracies_object[4]}NV"
        plt.plot(accuracies_object[0]*100, accuracies_object[1]*100, label=label, linewidth=2)

    plt.xlabel(x_label, font=fpath, fontsize=20)
    plt.ylabel("Mean accuracy (%)", font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.xticks(np.arange(0, 101, step=10), font=fpath, fontsize=15)
    plt.yticks(np.arange(0, 101, step=10), font=fpath, fontsize=15)

    legend = plt.legend(fontsize=15)
    plt.setp(legend.texts, font=fpath)

    if filename != "":
        plt.savefig(
            str(Path.home().joinpath("worsecrossbars", "outputs", "plots", "accuracies", filename)),
            dpi=300)

    plt.title(title, font=fpath, fontsize=20)
    plt.show()
