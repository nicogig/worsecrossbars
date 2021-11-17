"""
curves_plotting:
A plotting module used to generate training/validation and accuracy curves.
"""
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def load_curves_data(files, folder, curves_type):
    """
    This function loads the data that the user has elected to plot.

    Args:
      files: List containing tuples of the parameters of the files that are to be plotted. The
        structure should be [("STUCKZERO, "1", "0.5"), ..., ("STUCKHRS", "2", "3.1")].
      folder: Path string, employed to indicate the location from which to load files.
      curves_type: String indicating whether training/validation or accuracy data is desired.

    data_list:
      List containing the required data objects (either accuracies_objects or
      training_validation_objects).
    """

    data_list = []

    for filetuple in files:

        if curves_type == "accuracy":
            filepath = str(
                Path.home().joinpath(
                    "worsecrossbars",
                    "outputs",
                    folder,
                    "accuracies",
                    f"accuracies_{filetuple[0]}_{filetuple[1]}HL_"
                    + f"{float(filetuple[2])}NV.pickle",
                )
            )

        elif curves_type == "training_validation":
            filepath = str(
                Path.home().joinpath(
                    "worsecrossbars",
                    "outputs",
                    folder,
                    "training_validation",
                    f"training_validation_{filetuple[0]}_{filetuple[1]}HL_"
                    + f"{float(filetuple[2])}NV.pickle",
                )
            )

        else:
            raise ValueError(
                '"curves_type" parameter must be either "accuracy" or ' + '"training_validation".'
            )

        try:
            with open(filepath, "rb") as file:
                data_list.append(pickle.load(file))
        except FileNotFoundError as file_error:
            print(
                "The data you are attempting to plot does not exist. Please, check that the "
                + "command line arguments or the plots_data attribute in the .json file "
                + "are being entered in the format faultType_numberHiddenLayers_"
                + "noiseVariance, e.g. STUCKZERO_2_1.5."
            )
            raise file_error

    return data_list


def load_font():
    """
    This function loads the computern modern font used in the plots.

    fpath:
      Font path object pointing to the computern modern font.
    """

    if os.path.exists(Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf")):
        fpath = fm.FontProperties(
            fname=Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf"), size=18
        )
    else:
        fpath = fm.FontProperties(family="sans-serif", size=18)

    return fpath


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
        structure should be ["STUCKZERO_1_0.5", ..., "STUCKHRS"_"2"_"3.1"].
      folder: Path string, employed to indicate the location in which to save the plots.
      **kwargs: Valid keyword arguments are listed below
        - xlabel: String, label used on the x axis.
        - title: String, title used for the plot.
        - filename: String, name used to save the plot to file. If it is not provided (or is not a
            string), the plot is not saved to file.
    """

    # Turning list of strings into a list of tuples
    files = [tuple(filestring.split("_")) for filestring in files]

    # Importing LaTeX font for plots
    fpath = load_font()

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
        raise ValueError('"filename" parameter must be a valid string.')

    # Loading data
    accuracies_objects_list = load_curves_data(files, folder, "accuracy")

    # Plotting
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    for accuracies_object in accuracies_objects_list:
        label = f"{accuracies_object[2]}, {accuracies_object[3]}HL, {float(accuracies_object[4])}NV"
        plt.plot(
            accuracies_object[0] * 100,
            accuracies_object[1] * 100,
            label=label,
            linewidth=2,
        )

    plt.xlabel(xlabel, font=fpath, fontsize=20)
    plt.ylabel("Mean accuracy (%)", font=fpath, fontsize=20)
    plt.grid()
    plt.xticks(np.arange(0, 101, step=10), font=fpath, fontsize=14)
    plt.yticks(np.arange(0, 101, step=10), font=fpath, fontsize=14)

    accuracy_legend = plt.legend()
    plt.setp(accuracy_legend.texts, font=fpath, fontsize=14)

    if filename != "":
        plt.tight_layout()
        plt.savefig(
            str(
                Path.home().joinpath(
                    "worsecrossbars", "outputs", folder, "plots", "accuracies", filename
                )
            ),
            dpi=300,
        )
        return

    plt.title(title, font=fpath, fontsize=20)
    plt.show()


def training_validation_curves(files, folder, **kwargs):
    """
    This function plots training/validation curves with the given data.

    As a reminder, the following statements are true about the training_validation_objects contained
    in the training_validation_objects_list variable loaded below:
        training_accuracy_values = training_validation_object[0][0]
        validation_accuracy_values = training_validation_object[0][1]
        training_loss_values = training_validation_object[0][2]
        validation_loss_values = training_validation_object[0][3]
        fault_type = training_validation_object[1]
        number_hidden_layers = training_validation_object[2]
        noise_variance = training_validation_object[3]

    Args:
      files: List containing strings of the parameters of the files that are to be plotted. The
        structure should be ["STUCKZERO_1_0.5", ..., "STUCKHRS"_"2"_"3.1"].
      folder: Path string, employed to indicate the location in which to save the plots.
      **kwargs: Valid keyword arguments are listed below
        - title: String, title used for the plot.
        - filename: String, name used to save the plot to file. If it is not provided (or is not a
            string), the plot is not saved to file.
        - value_type: String indicating whether an accuracy plot or a loss plot is desired.
    """

    # Turning list of strings into a list of tuples
    files = [tuple(filestring.split("_")) for filestring in files]

    # Importing LaTeX font for plots
    fpath = load_font()

    # kwargs unpacking
    title = kwargs.get("title", "")
    filename = kwargs.get("filename", "")
    value_type = kwargs.get("value_type", "")

    # Validating arguments
    if not isinstance(value_type, str):
        raise ValueError('"value_type" parameter must be a string object.')

    if value_type.lower() != "accuracy" and value_type.lower() != "loss":
        raise ValueError('"value_type" parameter must be either "accuracy" or "loss".')

    if not isinstance(title, str):
        raise ValueError('"title" parameter must be a string object.')

    if not isinstance(filename, str):
        raise ValueError('"filename" parameter must be a valid string.')

    # Loading data
    training_validation_objects_list = load_curves_data(files, folder, "training_validation")

    epochs = list(range(1, len(training_validation_objects_list[0][0][0]) + 1))

    # Plotting
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    for training_validation_object in training_validation_objects_list:

        training_label = (
            f"Training ({training_validation_object[1]}, "
            + f"{training_validation_object[2]}HL, "
            + f"{float(training_validation_object[3])}NV)"
        )
        validation_label = (
            f"Validation ({training_validation_object[1]}, "
            + f"{training_validation_object[2]}HL, "
            + f"{float(training_validation_object[3])}NV)"
        )

        if value_type.lower() == "accuracy":

            if title == "":
                title = "Training/validation accuracy"
            ylabel = "Accuracy (%)"

            # Plotting training accuracy
            plt.plot(
                epochs,
                training_validation_object[0][0] * 100,
                "-o",
                markersize=7,
                label=training_label,
                linewidth=2,
            )

            # Plotting validation accuracy
            plt.plot(
                epochs,
                training_validation_object[0][1] * 100,
                "-D",
                markersize=7,
                label=validation_label,
                linewidth=2,
            )

        else:

            if title == "":
                title = "Training/validation loss"
            ylabel = "Loss"

            # Plotting training loss
            plt.plot(
                epochs,
                training_validation_object[0][2],
                "-o",
                markersize=7,
                label=training_label,
                linewidth=2,
            )

            # Plotting validation loss
            plt.plot(
                epochs,
                training_validation_object[0][3],
                "-D",
                markersize=7,
                label=validation_label,
                linewidth=2,
            )

    plt.xlabel("Epochs", font=fpath, fontsize=20)
    plt.ylabel(ylabel, font=fpath, fontsize=20)
    plt.grid()
    plt.xticks(np.arange(1, len(epochs) + 1, step=1), font=fpath, fontsize=14)
    plt.yticks(font=fpath, fontsize=14)

    training_validation_legend = plt.legend()
    plt.setp(training_validation_legend.texts, font=fpath, fontsize=14)

    if filename != "":
        plt.tight_layout()
        plt.savefig(
            str(
                Path.home().joinpath(
                    "worsecrossbars",
                    "outputs",
                    folder,
                    "plots",
                    "training_validation",
                    filename,
                )
            ),
            dpi=300,
        )
        return

    plt.title(title, font=fpath, fontsize=20)
    plt.show()


if __name__ == "__main__":

    # Command line parser for input arguments
    parser = argparse.ArgumentParser()

    # Adding required positional arguments to the command line parser. It should be noted that each
    # string in the "files" list has the structure faultType_numberHiddenLayers_noiseVariance,
    # e.g. STUCKZERO_1_0

    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILES",
        help="List of strings, each containing the parameters of a datafile to plot",
        type=str,
    )
    parser.add_argument(
        "--folder",
        metavar="FOLDER",
        required=True,
        help="String indicating the folder the user wants to load from / save to",
        type=str,
    )
    parser.add_argument(
        "--curves_type",
        metavar="CURVES_TYPE",
        required=True,
        help="String indicating whether training/validation or accuracy plots are desired",
        type=str,
        choices=["accuracy", "training_validation"],
    )

    # Adding optional arguments to the command line parser
    parser.add_argument(
        "-x",
        dest="xlabel",
        metavar="PLOT_XLABEL",
        help="Give the plot a specific xlabel",
        type=str,
        default="",
    )
    parser.add_argument(
        "-t",
        dest="title",
        metavar="PLOT_TITLE",
        help="Give the plot a specific title",
        type=str,
        default="",
    )
    parser.add_argument(
        "-f",
        dest="filename",
        metavar="PLOT_FILENAME",
        help="Give the plot a specific filename",
        type=str,
        default="",
    )
    parser.add_argument(
        "-v",
        dest="value_type",
        metavar="PLOT_VALUETYPE",
        help="Choose whether an accuracy or a loss plot should be produced",
        type=str,
        default="",
    )

    command_line_args = parser.parse_args()

    if command_line_args.curves_type == "accuracy":
        accuracy_curves(
            command_line_args.files,
            command_line_args.folder,
            xlabel=command_line_args.xlabel,
            title=command_line_args.title,
            filename=command_line_args.filename,
        )

    elif command_line_args.curves_type == "training_validation":
        training_validation_curves(
            command_line_args.files,
            command_line_args.folder,
            title=command_line_args.title,
            filename=command_line_args.filename,
            value_type=command_line_args.value_type,
        )
