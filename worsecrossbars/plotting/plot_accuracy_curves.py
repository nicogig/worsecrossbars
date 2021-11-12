"""
plot_accuracy_curves:
A plotting module used to generate accuracy curves.
"""

import os
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

    for filename in command_line_args.files:

        with open(str(Path.home().joinpath("worsecrossbars", "outputs", "accuracies", filename)),
                  "rb") as file:
            accuracies_objects_list.append(pickle.load(file))

    accuracy_curves_plotter(accuracies_objects_list, fpath=font, x_label=command_line_args.xlabel,
                            title=command_line_args.title, filename=command_line_args.filename)


if __name__ == "__main__":

    # Command line parser for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", dest="title", metavar="PLOT_TITLE",
        help="Give the plot a specific title", type=str, default="")
    parser.add_argument("-x", dest="xlabel", metavar="PLOT_XLABEL",
        help="Give the plot a specific xlabel", type=str, default="")
    parser.add_argument("--filename", dest="filename", metavar="PLOT_FILENAME",
        help="Give the plot a specific filename", type=str, default="")
    parser.add_argument("--files", dest="files", metavar="FILES",
        help="List of strings with names of files to add to the plot", type=list, default=None)

    command_line_args = parser.parse_args()

    # Importing LaTeX font for plots
    if os.path.exists(Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf")):
        font = fm.FontProperties(
            fname=Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf"),
            size=18)
    else:
        font = fm.FontProperties(family="sans-serif", size=18)

    # GoTo main
    main()
