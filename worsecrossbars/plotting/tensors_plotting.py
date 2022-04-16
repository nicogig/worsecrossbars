import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


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


def plot_tensor(figname: str, filepath: Path, xlabel: str):
    """"""

    y_label = "Absolute frequency"

    fpath = load_font()
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    items = []

    with open(filepath) as f:
        lines = f.readlines()

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

    for file in zip(filenames, fignames, xlabels):

        filepath = Path.home().joinpath("worsecrossbars", "tensors", file[0])
        plot_tensor(figname=file[1], filepath=filepath, xlabel=file[2])
