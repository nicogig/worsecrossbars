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
import numpy as np
import matplotlib.pyplot as plt


def accuracy_curves(files, folder, **kwargs):
    """
    This function plots accuracy curves with the given data.

    As a reminder, the following statements are true about the accuracies_objects contained in the
    accuracies_objects_list variable loaded below:
        percentages = accuracies_object[0]
        accuracies = accuracies_object[1]
        fault_type = accuracies_object[2]
        number_hidden_layers = accuracies_object[3]
        noise_variance = accuracies_object[4]

    Args:
      files: List containing strings of the parameters of the files that are to be plotted. The
        structure should be ["STUCKZERO_1_0.5", ..., "STUCKHRS"_"2"_"3.1"]
      folder: Path string, employed to both load files and save plots
      **kwargs: Valid keyword arguments are listed below
        xlabel: String, label used on the x axis.
        title: String, title used for the plot.
        filename: String, name used to save the plot to file. If it is not provided (or is not a
          string), the plot is not saved to file.
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
    xlabel = kwargs.get("xlabel", "")
    title = kwargs.get("title", "")
    filename = kwargs.get("filename", "")

    # Validating arguments
    if not isinstance(xlabel, str) or xlabel == "":
        xlabel = "Percentage of faulty devices (%)"

    if not isinstance(title, str) or title == "":
        title = "Influence of faulty devices on ANN inference accuracy"

    if not isinstance(filename, str):
        raise ValueError("\"filename\" parameter must be a valid string.")

    # Loading data
    accuracies_objects_list = []

    for filetuple in files:

        try:
            with open(str(Path.home().joinpath("worsecrossbars", "outputs", folder,
            "accuracies", f"accuracies_{filetuple[0]}_{filetuple[1]}HL_{filetuple[2]}NV.pickle")),
            "rb") as file:
                accuracies_objects_list.append(pickle.load(file))
        except FileNotFoundError as e:
            print("The data you are attempting to plot does not exist. Please, check that the " +
                  "command line arguments or the plots_data attribute in the .json file " +
                  "are being entered in the format faultType_numberHiddenLayers_" +
                  "noiseVariance, e.g. STUCKZERO_2_1.5.")
            raise e

    # Plotting
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    for accuracies_object in accuracies_objects_list:
        label = f"{accuracies_object[2]}, {accuracies_object[3]}HL, {accuracies_object[4]}NV"
        plt.plot(accuracies_object[0]*100, accuracies_object[1]*100, label=label, linewidth=2)

    plt.xlabel(xlabel, font=fpath, fontsize=20)
    plt.ylabel("Mean accuracy (%)", font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.xticks(np.arange(0, 101, step=10), font=fpath, fontsize=15)
    plt.yticks(np.arange(0, 101, step=10), font=fpath, fontsize=15)

    legend = plt.legend(fontsize=15)
    plt.setp(legend.texts, font=fpath)

    if filename != "":
        plt.savefig(
            str(Path.home().joinpath("worsecrossbars", "outputs", folder,
                                     "plots", "accuracies", filename)), dpi=300)

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
    parser.add_argument("-x", dest="xlabel", metavar="PLOT_XLABEL",
        help="Give the plot a specific xlabel", type=str, default="")
    parser.add_argument("-t", dest="title", metavar="PLOT_TITLE",
        help="Give the plot a specific title", type=str, default="")
    parser.add_argument("-f", dest="filename", metavar="PLOT_FILENAME",
        help="Give the plot a specific filename", type=str, default="")

    command_line_args = parser.parse_args()

    accuracy_curves(command_line_args.files, command_line_args.folder,
                    xlabel=command_line_args.xlabel, title=command_line_args.title,
                    filename=command_line_args.filename)
