"""
training_validation_curves_plotting:
A plotting module used to generate training/validation curves.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import matplotlib.font_manager as fm

# def training_validation_plotter(objects_list, fpath=None, value_type="", title="", filename=""):
#     """
#     """

#     if fpath is None:
#         warnings.warn("Remember to pass an \"fpath\" parameter to control the plot's font.")
    
#     if not isinstance(value_type, str):
#         raise ValueError("\"value_type\" parameter must be a string object.")

#     if (value_type.lower() != "accuracy" and value_type.lower() != "loss"):
#         raise ValueError("\"value_type\" parameter must be either \"Accuracy\" or \"Loss\".")

#     if not isinstance(title, str) or title == "":
#         title = "Training/validation curves"

#     if not isinstance(filename, str):
#         raise ValueError("\"filename\" parameter must be a valid string.")

#     fig = plt.figure()
#     fig.set_size_inches(10, 6)

#     for training_validation_object in objects_list:

#         label = f"{training_validation_object[1]}, {training_validation_object[2]}HL," + \
#                 f" {training_validation_object[3]}NV"

#         if value_type.lower() == "accuracy":

#             pass

#         else:

#             pass

#         # plt.plot(accuracies_object[0]*100, accuracies_object[1]*100, label=label, linewidth=2)

#     plt.xlabel("Epochs", font=fpath, fontsize=20)

#     plt.grid()
#     plt.tight_layout()
#     plt.xticks(np.arange(0, 101, step=10), font=fpath, fontsize=15)
#     plt.yticks(np.arange(0, 101, step=10), font=fpath, fontsize=15)

#     legend = plt.legend(fontsize=15)
#     plt.setp(legend.texts, font=fpath)

#     if filename != "":
#         plt.savefig(
#             str(Path.home().joinpath("worsecrossbars", "outputs", "plots", "accuracies", filename)),
#             dpi=300)

#     plt.title(title, font=fpath, fontsize=20)
#     plt.show()


# #     epochs,
# #     training,
# #     validation,
# #     value_type="",
# #     number_hidden_layers=None,
# #     fpath=None,
# #     save=False,
# #     label_step=1):
# #     """
# #     training_validation_plotter:
# #     Plot the Training and Validation curves with respect to the Epochs.
# #     Inputs:
# #         -   epochs: The range of the epochs used to train the network.
# #         -   training: An array (list) containing the values obtained from the training stage.
# #         -   validation: An array (list) containing the values obtained from the validation stage.
# #         -   value_type: A string describing the type of data.
# #         -   Allowed values are "Accuracy" and "Loss".
# #         -   number_hidden_layers: An integer indicating which network topolgy is being examined.
# #         -   fpath: The FontProperties object containing information about the font to use.
# #         -   save: Boolean specifying whether the plot is to be saved to a file.
# #         -   label_step: The step for the labels on the x-axis. Defaults to 1.
# #     Output:
# #         -   A graph, both inline (if using Jupyter)
# #         -   and as a png (if the "save" flag is set to true).
# #     """

# #     if (value_type.lower() != "accuracy" and value_type.lower() != "loss"):
# #         raise ValueError("\"value_type\" parameter must be either \"Accuracy\" or \"Loss\".")

# #     if number_hidden_layers is None:
# #         raise ValueError("\"number_hidden_layers\" parameter must" + \
# #             " be passed to the plotting function.")

# #     if fpath is None:
# #         raise ValueError("\"fpath\" parameter must be passed to the plotting function.")

# #     fig = plt.figure()
# #     fig.set_size_inches(10, 6)

# #     title = f"Training and validation {value_type.lower()}, {number_hidden_layers}HL"

# #     if value_type.lower() == "accuracy":
# #         y_label = f"{value_type.title()} (%)"
# #         training_legend_label = f"Training {value_type.lower()} (%)"
# #         validation_legend_label = f"Validation {value_type.lower()} (%)"
# #         plt.plot(
# #             epochs,
# #             training*100,
# #             "-bo",
# #             markersize=7,
# #             label=training_legend_label,
# #             linewidth=2)
# #         plt.plot(
# #             epochs,
# #             validation*100,
# #             "-rD",
# #             markersize=7,
# #             label=validation_legend_label,
# #             linewidth=2)
# #         legend = plt.legend(fontsize=15, loc="lower right")
# #     else:
# #         y_label = value_type.title()
# #         training_legend_label = f"Training {value_type.lower()}"
# #         validation_legend_label = f"Validation {value_type.lower()}"
# #         plt.plot(epochs, training, "-bo", markersize=7, label=training_legend_label, linewidth=2)
# #         plt.plot(
# #             epochs,
# #             validation,
# #             "-rD",
# #             markersize=7,
# #             label=validation_legend_label,
# #             linewidth=2)
# #         legend = plt.legend(fontsize=15, loc="upper right")

# #     plt.xlabel("Epochs", font=fpath, fontsize=20)
# #     plt.ylabel(y_label, font=fpath, fontsize=20)
# #     plt.grid()
# #     plt.tight_layout()
# #     plt.xticks(np.arange(0, len(epochs) + 1, step=label_step), font=fpath, fontsize=15)
# #     plt.yticks(font=fpath, fontsize=15)

# #     plt.setp(legend.texts, font=fpath)

# #     if save:
# #         plt.savefig(
# #             str(Path.home().joinpath("worsecrossbars",
# #             "outputs",
# #             "plots",
# #             "training_validation",
# #             f"training_validation_{value_type.lower()}_plot_{number_hidden_layers}HL.png")),
# #             dpi=200)

# #     plt.title(title, font=fpath, fontsize=20)
# #     plt.show()



def training_validation_curves():
    """
    This function generates training/validation curves with the given data.
    """

    training_validation_objects_list = []

    for filetuple in command_line_args.files:

        try:
            with open(str(Path.home().joinpath("worsecrossbars", "outputs", "training_validation",
            f"training_validation_{filetuple[0]}_{filetuple[1]}HL_{filetuple[2]}NV.pickle")),
            "rb") as file:
                training_validation_objects_list.append(pickle.load(file))
        except FileNotFoundError:
            print("The data you are attempting to plot does not exist. Please, check the command " +
                  "line arguments are being entered in the format faultType_numberHiddenLayers_" +
                  "noiseVariance, e.g. STUCKZERO_1_0.")
            sys.exit(1)

    #training_validation_plotter(training_validation_objects_list, fpath=font,
                                #value_type=command_line_args.value_type,
                                #title=command_line_args.title, filename=command_line_args.filename)


if __name__ == "__main__":

    # Command line parser for input arguments
    parser = argparse.ArgumentParser()

    # Adding a required positional argument to the command line parser. It should be noted that each
    # string in the list below has the structure faultType_numberHiddenLayers_noiseVariance,
    # e.g. STUCKZERO_1_0

    parser.add_argument("files", nargs="+", metavar="FILES",
        help="List of strings, each containing the parameters of a datafile to plot", type=str)

    # Adding optional arguments to the command line parser
    parser.add_argument("-t", dest="title", metavar="PLOT_TITLE",
        help="Give the plot a specific title", type=str, default="")
    parser.add_argument("-x", dest="xlabel", metavar="PLOT_XLABEL",
        help="Give the plot a specific xlabel", type=str, default="")
    parser.add_argument("-f", dest="filename", metavar="PLOT_FILENAME",
        help="Give the plot a specific filename", type=str, default="")

    command_line_args = parser.parse_args()

    # Turning list of strings in command_line_args.files into a list of tuples
    command_line_args.files = [tuple(filestring.split("_")) for filestring in
                               command_line_args.files]

    # Importing LaTeX font for plots
    if os.path.exists(Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf")):
        font = fm.FontProperties(
            fname=Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf"),
            size=18)
    else:
        font = fm.FontProperties(family="sans-serif", size=18)

    # GoTo main
    training_validation_curves()
