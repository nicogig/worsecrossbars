import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from worsecrossbars.utilities.io_operations import read_external_json


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        metavar="files",
        type=str,
        nargs="+",
        help="List of files to be plotted.",
    )
    parser.add_argument(
        "--x_data",
        type=float,
        nargs="+",
        metavar="x_data",
        help="List of x-data to be plotted.",
    )
    parser.add_argument(
        "--x_label",
        type=str,
        metavar="x_label",
        help="Label for the x-axis.",
    )
    parser.add_argument(
        "--y_label",
        type=str,
        metavar="y_label",
        help="Label for the y-axis.",
    )

    fpath = load_font()
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    y_data = []
    x_data = parser.parse_args().x_data

    for f in parser.parse_args().files:
        json_object = read_external_json(str(Path.cwd().joinpath("".join(f))))
        y_data.append(np.array(json_object["accuracies"]) * 100)

    print(y_data)
    print(x_data)

    plt.plot(x_data, y_data, linewidth=2)
    plt.xlabel(parser.parse_args().x_label, fontproperties=fpath)
    plt.ylabel(parser.parse_args().y_label, fontproperties=fpath)
    plt.grid()
    plt.tight_layout()

    plt.savefig("plot.png", dpi=300)
