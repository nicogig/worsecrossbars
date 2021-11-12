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
import numpy as np
import matplotlib.pyplot as plt


def training_validation_curves(files, folder, **kwargs):
    """
    This function plots training/validation curves with the given data.

    Args:
      files: List containing strings of the parameters of the files that are to be plotted. The
        structure should be ["STUCKZERO_1_0.5", ..., "STUCKHRS"_"2"_"3.1"]
      folder: Path string, employed to both load files and save plots
      **kwargs: Valid keyword arguments are listed below
        title: String, title used for the plot.
        filename: String, name used to save the plot to file. If it is not provided (or is not a
          string), the plot is not saved to file.
        value_type: String indicating weather an accuracy plot or a loss plot is desired
    """

    # Turning list of strings into a list of tuples
    files = [tuple(filestring.split("_")) for filestring in files]

    # Importing LaTeX font for plots
    if os.path.exists(Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf")):
        fpath = fm.FontProperties(
            fname=Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf"),
            size=18)
    else:
        fpath = fm.FontProperties(family="sans-serif", size=18)

    # kwargs unpacking
    title = kwargs.get("title", "")
    filename = kwargs.get("filename", "")
    value_type = kwargs.get("value_type", "")

    # Validating arguments
    if not isinstance(value_type, str):
        raise ValueError("\"value_type\" parameter must be a string object.")

    if (value_type.lower() != "accuracy" and value_type.lower() != "loss"):
        raise ValueError("\"value_type\" parameter must be either \"Accuracy\" or \"Loss\".")

    if not isinstance(title, str):
        raise ValueError("\"title\" parameter must be a string object.")

    if not isinstance(filename, str):
        raise ValueError("\"filename\" parameter must be a valid string.")

    # Loading data
    training_validation_objects_list = []

    for filetuple in files:

        try:
            with open(str(Path.home().joinpath("worsecrossbars", "outputs", folder,
            "training_validation", f"training_validation_{filetuple[0]}_{filetuple[1]}HL_",
            f"{filetuple[2]}NV.pickle")), "rb") as file:
                training_validation_objects_list.append(pickle.load(file))
        except FileNotFoundError as e:
            print("The data you are attempting to plot does not exist. Please, check that the " +
                  "command line arguments or the plots_data attribute in the .json file " +
                  "are being entered in the format faultType_numberHiddenLayers_" +
                  "noiseVariance, e.g. STUCKZERO_2_1.5.")
            raise e
    
    epochs = list(range(1, len(training_validation_objects_list[0][0][0])+1))

    # Plotting
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    for training_validation_object in training_validation_objects_list:

        training_label = f"Training ({training_validation_object[1]}, " + \
                             f"{training_validation_object[2]}HL, " + \
                             f"{training_validation_object[3]}NV)"
        validation_label = f"Validation ({training_validation_object[1]}, " + \
                             f"{training_validation_object[2]}HL, " + \
                             f"{training_validation_object[3]}NV)"

        if value_type.lower() == "accuracy":

            if title == "":
                title = "Training/validation accuracy"
            ylabel = "Accuracy (%)"

            # Plotting training accuracy
            plt.plot(epochs, training_validation_object[0][0]*100, "-bo", markersize=7,
                     label=training_label, linewidth=2)

            # Plotting validation accuracy
            plt.plot(epochs, training_validation_object[0][1]*100, "-rD", markersize=7,
                     label=validation_label, linewidth=2)

        else:

            if title == "":
                title = "Training/validation loss"
            ylabel = "Loss"

            # Plotting training loss
            plt.plot(epochs, training_validation_object[0][2], "-bo", markersize=7,
                     label=training_label, linewidth=2)

            # Plotting validation loss
            plt.plot(epochs, training_validation_object[0][3], "-rD", markersize=7,
                     label=validation_label, linewidth=2)            

    plt.xlabel("Epochs", font=fpath, fontsize=20)
    plt.ylabel(ylabel, font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.xticks(np.arange(1, len(epochs) + 1, step=1), font=fpath, fontsize=15)
    plt.yticks(font=fpath, fontsize=15)

    legend = plt.legend(fontsize=15)
    plt.setp(legend.texts, font=fpath)

    if filename != "":
        plt.savefig(
            str(Path.home().joinpath("worsecrossbars", "outputs", folder,
                                     "plots", "training_validation", filename)), dpi=300)

    plt.title(title, font=fpath, fontsize=20)
    plt.show()


if __name__ == "__main__":

    # Command line parser for input arguments
    parser = argparse.ArgumentParser()

    # Adding required positional arguments to the command line parser. It should be noted that each
    # string in the list below has the structure faultType_numberHiddenLayers_noiseVariance,
    # e.g. STUCKZERO_1_0

    parser.add_argument("files", nargs="+", metavar="FILES",
        help="List of strings, each containing the parameters of a datafile to plot", type=str)
    parser.add_argument("--folder", metavar="FOLDER", required=True,
        help="String indicating the folder the user wants to load from / save to", type=str)

    # Adding optional arguments to the command line parser
    parser.add_argument("-t", dest="title", metavar="PLOT_TITLE",
        help="Give the plot a specific title", type=str, default="")
    parser.add_argument("-f", dest="filename", metavar="PLOT_FILENAME",
        help="Give the plot a specific filename", type=str, default="")
    parser.add_argument("-v", dest="value_type", metavar="PLOT_VALUETYPE",
        help="Choose whether an accuracy or a loss plot should be produced", type=str, default="")

    command_line_args = parser.parse_args()

    training_validation_curves(command_line_args.files, command_line_args.folder,
                               title=command_line_args.title, filename=command_line_args.filename,
                               value_type=command_line_args.value_type)
