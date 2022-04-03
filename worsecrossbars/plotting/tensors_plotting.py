import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

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

def plot_tensor(filename: str, filepath: str):
    """"""

    x_label = "Weight"
    y_label = "Absolute Frequency"

    fpath = load_font()
    fig = plt.figure()
    fig.set_size_inches(10, 6)

    items = []

    with open(filepath, "r") as f:
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

    items = np.array(items)
    print(f"Shape: {items.shape}")
    print(f"Mean: {items.mean()}")
    print(f"Std: {items.std()}")
    print(f"Max: {items.max()}")
    print(f"Min: {items.min()}")

    plt.hist(items, bins="auto")
    plt.xticks(font=fpath, fontsize=14)
    plt.yticks(font=fpath, fontsize=14)
    plt.xlabel(x_label, font=fpath, fontsize=20)
    plt.ylabel(y_label, font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)


if __name__ == "__main__":

    # filename = "weights.txt"
    # filename = "conductances_pre_alteration.txt"
    filename = "conductances_post_alteration.txt"
    # filename = "currents.txt"
    # filename = "y_disturbed.txt"
    filepath = Path.home().joinpath("worsecrossbars", "tensors", filename)
