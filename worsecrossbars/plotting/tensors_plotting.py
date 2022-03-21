import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_tensor(filename: str, filepath: str):
    """"""

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
    plt.savefig(f"{filename}.png", dpi=300)


if __name__ == "__main__":

    # filename = "weights.txt"
    # filename = "conductances_pre_alteration.txt"
    filename = "conductances_post_alteration.txt"
    # filename = "currents.txt"
    # filename = "y_disturbed.txt"
    filepath = Path.home().joinpath("worsecrossbars", "tensors", filename)
