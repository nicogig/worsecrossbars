import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from worsecrossbars.utilities.io_operations import read_external_json
from worsecrossbars.utilities.json_handlers import validate_json
from worsecrossbars.utilities.json_schemas import plot_schema


def load_font() -> FontProperties:
    """This function loads the computern modern font used in the plots.

    fpath:
      Font path object pointing to the computern modern font.
    """

    if os.path.exists(Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf")):
        fpath = FontProperties(
            fname=Path.home().joinpath("worsecrossbars", "utils", "cmunrm.ttf"), size=18
        )
    else:
        fpath = FontProperties(family="sans-serif", size=18)

    return fpath


def plot(json_object: dict):
    i = 0
    fpath = load_font()
    xlabel = "Percentage of faulty devices (%)"
    for plot in json_object["plots"]:
        fig = plt.figure()
        fig.set_size_inches(10, 6)
        for f in plot["files"]:
            file_as_json = read_external_json(str(Path.cwd().joinpath(f)))
            sim_parameters = file_as_json["simulation_parameters"]
            scaling = 1 / (len(file_as_json["accuracies"]) - 1)
            x_data = np.arange(0.0, 1.01, scaling).round(2)
            label = ""
            for feature in plot["key_features"]:
                if feature == "number_hidden_layers":
                    label += f"{sim_parameters[feature]} HL, "
                else:
                    label += f"{sim_parameters[feature]}, "
            if sim_parameters["discretisation"]:
                plt.plot(
                    x_data * 100,
                    np.array(file_as_json["pre_discretisation_accuracies"]) * 100,
                    label=label + " (Pre-Discretisation)",
                    linewidth=2,
                )
                plt.plot(
                    x_data * 100,
                    np.array(file_as_json["accuracies"]) * 100,
                    label=label + " (Discretised)",
                    linewidth=2,
                )
            else:
                plt.plot(
                    x_data * 100,
                    np.array(file_as_json["accuracies"]) * 100,
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
        plt.tight_layout()

        plt.savefig(f"plot_{i}.png", dpi=300)
        i += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        metavar="CONFIG_FILE",
        nargs="?",
        help="Provide the config file needed for plotting",
        type=str,
        default="",
    )

    command_line_args = parser.parse_args()
    json_path = Path.cwd().joinpath(command_line_args.config)
    json_object = read_external_json(str(json_path))

    validate_json(json_object, plot_schema)

    plot(json_object)
