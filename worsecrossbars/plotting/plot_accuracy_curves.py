"""
plot_accuracy_curves:
A plotting module used to generate accuracy curves.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import matplotlib.font_manager as fm
from worsecrossbars.plotting.plotting import accuracy_curves_plotter


def main():
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

    accuracy_curves_plotter(accuracies_objects_list, fpath=font, x_label=command_line_args.xlabel,
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

    # GoTo main
    main()
