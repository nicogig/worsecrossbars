"""
accuracy_curves_plotting:
A plotting module used to generate accuracy curves.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import matplotlib.font_manager as fm
import warnings
import numpy as np
import matplotlib.pyplot as plt


def accuracy_plotter(objects_list, fpath=None, x_label="", title="", filename=""):
    """
    This function plots accuracy curves with the given data.

    As a reminder, the following statements are true about the accuracies_objects contained in the
    objects_list argument:
        percentages = accuracies_object[0]
        accuracies = accuracies_object[1]
        fault_type = accuracies_object[2]
        number_hidden_layers = accuracies_object[3]
        noise_variance = accuracies_object[4]

    Args:
      objects_list: List containing the accuracies_objects that are to be plotted.
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

    for accuracies_object in objects_list:
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


def accuracy_curves(files, ):
    """
    This function generates accuracy curves with the given data.
    """

    accuracies_objects_list = []

    for filetuple in command_line_args.files:

        try:
            with open(str(Path.home().joinpath("worsecrossbars", "outputs", "accuracies",
            f"accuracies_{filetuple[0]}_{filetuple[1]}HL_{filetuple[2]}NV.pickle")), "rb") as file:
                accuracies_objects_list.append(pickle.load(file))
        except FileNotFoundError:
            print("The data you are attempting to plot does not exist. Please, check the command " +
                  "line arguments are being entered in the format faultType_numberHiddenLayers_" +
                  "noiseVariance, e.g. STUCKZERO_1_0.")
            sys.exit(1)

    accuracy_plotter(accuracies_objects_list, fpath=font, x_label=command_line_args.xlabel,
                     title=command_line_args.title, filename=command_line_args.filename)


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

    accuracy_curves()
