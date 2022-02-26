"""simulations:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""
import json

import numpy as np
import torch

from worsecrossbars.utilities.msteams_notifier import MSTeamsNotifier
from worsecrossbars.utilities.io_operations import read_webhook
from worsecrossbars.pytorch.nonidealities import StuckAtValue
from worsecrossbars.pytorch.memristive_mlp import MemristiveMLP
from worsecrossbars.pytorch.dataloaders import mnist_dataloaders


def _train_evaluate(
    G_off: float, G_on: float, k_V: float, nonideality: StuckAtValue, device: torch.device, **kwargs
):

    # Unpacking keyword arguments
    number_hidden_layers = kwargs.get("number_hidden_layers", 2)
    epochs = kwargs.get("epochs", 10)
    simulations = kwargs.get("simulations", 100)
    dataloaders = kwargs.get("dataloaders", mnist_dataloaders())

    average_accuracy = 0

    for idx in range(simulations):
        print(idx)
        model = MemristiveMLP(
            number_hidden_layers, G_off, G_on, k_V, nonidealities=[nonideality], device=device
        )
        model.compile("rmsprop")
        *_, test_accuracy = model.fit(dataloaders, epochs)

        average_accuracy += test_accuracy

    average_accuracy /= simulations

    return average_accuracy


def stuck_simulation(
    value: float, G_off: float, G_on: float, k_V: float, device: torch.device, **kwargs
):

    # Unpacking keyword arguments
    percentages = kwargs.get("percentages", np.arange(0, 1.01, 0.01).round(2))
    number_hidden_layers = kwargs.get("number_hidden_layers", 2)
    epochs = kwargs.get("epochs", 10)
    simulations = kwargs.get("simulations", 100)
    dataloaders = kwargs.get("dataloaders", mnist_dataloaders())

    accuracies = []

    # Maybe this could be done in parallel?
    # TODO
    for percentage in percentages:
        print(percentage)
        nonideality = StuckAtValue(value, percentage, device)
        accuracy = _train_evaluate(
            G_off,
            G_on,
            k_V,
            nonideality,
            device,
            number_hidden_layers=number_hidden_layers,
            epochs=epochs,
            simulations=simulations,
            dataloaders=dataloaders,
        )

        accuracies.append(accuracy)

    return accuracies


if __name__ == "__main__":

    """
    High Nonlinearity.
    G_off 7.723066346443375e-07
    G_on 2.730684400376049e-06
    n_avg 2.98889722498956
    n_std 0.36889602261470894
    Low Nonlinearity.
    G_off 0.0009971787221729755
    G_on 0.003513530595228076
    n_avg 2.132072652112917
    n_std 0.09531988936898476
    """
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.device(device)

    print(f"Running on {device.type}")

    teams = MSTeamsNotifier(read_webhook())

    dataloaders = mnist_dataloaders()

    G_off = 0.0009971787221729755
    G_on = 0.003513530595228076
    k_V = 0.5

    teams.send_message("Started Simulation", color="ffca33")
    accuracies = stuck_simulation(0, G_off, G_on, k_V, device, dataloaders=dataloaders)
    teams.send_message("Ended Simulation", color="ffca33")

    with open("accuracies.json", "w", encoding="utf-8") as f:
        json.dump(accuracies, f, ensure_ascii=False, indent=4)
