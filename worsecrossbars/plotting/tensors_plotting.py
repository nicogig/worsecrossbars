"""tensor_plotting:
A module used to plot key tensors obtained from the simulations.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from worsecrossbars.utilities.load_font import load_font


def plot_tensor(figname: str, filepath: Path, xlabel: str) -> None:
    """This function plots the tensors obtained from the simulation.

    Args:
      figname: Name with which to save the produced figure.
      filepath: Path to the file storing tensor contents.
      xlabel: Label for the x-axis.
    """

    y_label = "Absolute frequency"

    fpath = load_font()
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    items = []

    # pylint: disable=unspecified-encoding
    with open(filepath) as file:
        lines = file.readlines()

    # pylint: enable=unspecified-encoding

    for line in lines:

        items.extend(
            [
                float(item)
                for item in line.replace("[", "")
                .replace("]", "")
                .replace("\n", "")
                .replace(",", " ")
                .strip()
                .split(" ")
            ]
        )

    items_arr = np.array(items)
    print(f"Shape: {items_arr.shape}")
    print(f"Mean: {items_arr.mean()}")
    print(f"Std: {items_arr.std()}")
    print(f"Max: {items_arr.max()}")
    print(f"Min: {items_arr.min()}")

    if xlabel == "Conductance (μS)":
        items_arr *= 10 ** 6

    plt.hist(items_arr, bins="auto")
    plt.xticks(font=fpath, fontsize=14)
    plt.yticks(font=fpath, fontsize=14)
    plt.xlabel(xlabel, font=fpath, fontsize=20)
    plt.ylabel(y_label, font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{figname}.png", dpi=300)


if __name__ == "__main__":

    filenames = [
        "weights.txt",
        "conductances_pre_alteration.txt",
        "conductances_post_alteration.txt",
        "currents.txt",
        "y_disturbed.txt",
    ]
    fignames = [
        "weights_5epochs_100faulty",
        "cond_pre_5epochs_100faulty",
        "cond_post_5epochs_100faulty",
        "currents_5epochs_100faulty",
        "y_disturbed_5epochs_100faulty",
    ]
    xlabels = ["Weight", "Conductance (μS)", "Conductance (μS)", "Current (A)", "Layer output"]

    for file_tuple in zip(filenames, fignames, xlabels):

        path = Path.home().joinpath("worsecrossbars", "tensors", file_tuple[0])
        plot_tensor(figname=file_tuple[1], filepath=path, xlabel=file_tuple[2])
