"""simulation:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""
import json
from typing import Tuple

import numpy as np
import tensorflow as tf
from numpy import ndarray

from worsecrossbars.backend.mlp_generator import mnist_mlp
from worsecrossbars.backend.mlp_trainer import train_mlp
from worsecrossbars.backend.nonidealities import D2DVariability
from worsecrossbars.backend.nonidealities import IVNonlinear
from worsecrossbars.backend.nonidealities import StuckAtValue
from worsecrossbars.backend.nonidealities import StuckDistribution


def _simulate(
    simulation_parameters: dict,
    nonidealities: list,
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    batch_size: int = 100,
    horovod: bool = False,
) -> Tuple[float, float]:
    """"""

    simulation_accuracies = np.zeros(simulation_parameters["number_simulations"])
    pre_discretisation_simulation_accuracies = np.zeros(simulation_parameters["number_simulations"])

    discretise = simulation_parameters["number_conductance_levels"] > 0

    for simulation in range(simulation_parameters["number_simulations"]):

        nonideality_labels = [nonideality.label() for nonideality in nonidealities]
        print(f"Simulation #{simulation+1}, nonidealities: {nonideality_labels}")

        model = mnist_mlp(
            simulation_parameters["G_off"],
            simulation_parameters["G_on"],
            simulation_parameters["k_V"],
            nonidealities=nonidealities,
            number_hidden_layers=simulation_parameters["number_hidden_layers"],
            noise_variance=simulation_parameters["noise_variance"],
            horovod=horovod,
        )

        mlp_weights, mlp_history, pre_discretisation_accuracy = train_mlp(
            dataset,
            model,
            epochs=60,
            batch_size=batch_size,
            discretise=discretise,
            hrs_lrs_ratio=simulation_parameters["G_on"] / simulation_parameters["G_off"],
            number_conductance_levels=simulation_parameters["number_conductance_levels"],
            excluded_weights_proportion=simulation_parameters["excluded_weights_proportion"],
            horovod=horovod,
        )

        with open(f"mlp_weights_{simulation+1}.json", "w") as file:
            out = []
            for arr in mlp_weights:
                out.append(arr.tolist())
            json.dump(out, file)

        simulation_accuracies[simulation] = model.evaluate(dataset[1][0], dataset[1][1])[1]
        pre_discretisation_simulation_accuracies[simulation] = pre_discretisation_accuracy

    average_accuracy = simulation_accuracies.mean()
    average_pre_discretisation_accuracy = pre_discretisation_simulation_accuracies.mean()

    return average_accuracy, average_pre_discretisation_accuracy


def run_simulations(
    simulation_parameters: dict,
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    batch_size: int = 100,
    horovod: bool = False,
) -> Tuple[ndarray, ndarray]:
    """This function...

    Returns:
      accuracies: Numpy ndarray containing one (or multiple) float value(s),
      pre_discretisation_accuracies: Numpy ndarray containing one (or multiple) float value(s),
    """

    # Generating list of nonidealities
    nonidealities = []

    for nonideality in simulation_parameters["nonidealities"]:

        if nonideality["type"] == "IVNonlinear":
            nonidealities.append(
                IVNonlinear(
                    V_ref=nonideality["parameters"][0],
                    avg_gamma=nonideality["parameters"][1],
                    std_gamma=nonideality["parameters"][2],
                )
            )
            simulation_parameters["nonidealities"].remove(nonideality)

        elif nonideality["type"] == "D2DVariability":
            nonidealities.append(
                D2DVariability(
                    simulation_parameters["G_off"],
                    simulation_parameters["G_on"],
                    nonideality["parameters"][0],
                    nonideality["parameters"][1],
                )
            )
            simulation_parameters["nonidealities"].remove(nonideality)

    if not simulation_parameters["nonidealities"]:

        # If no other nonidealities (i.e. no device-percentage-based nonidealities remain), there
        # is no need to simualate varying percentages of faulty devices.
        simulation_results = _simulate(
            simulation_parameters, nonidealities, dataset, batch_size=batch_size, horovod=horovod
        )
        accuracies = np.array([simulation_results[0]])
        pre_discretisation_accuracies = np.array([simulation_results[1]])

        return accuracies, pre_discretisation_accuracies

    # Generating vectors to be used for device-percentage-based nonidealities analysis
    percentages = np.arange(0, 1.01, 0.02).round(2)
    accuracies = np.zeros(percentages.size)
    pre_discretisation_accuracies = np.zeros(percentages.size)

    # Adding percentage-based nonidealities
    for nonideality in simulation_parameters["nonidealities"]:

        if nonideality["type"] == "StuckAtValue":
            nonidealities.append(StuckAtValue(value=nonideality["parameters"][0]))
            simulation_parameters["nonidealities"].remove(nonideality)

        elif nonideality["type"] == "StuckDistribution":
            # If only nonideality["parameters"][0] is a single integer, then this indicates the
            # number of weights to create between G_off and G_on. Otherwise, if
            # nonideality["parameters"][0] is a list, this is passed as distrib to
            # StuckDistribution()
            if isinstance(nonideality["parameters"][0], list):
                nonidealities.append(StuckDistribution(distrib=nonideality["parameters"][0]))
                simulation_parameters["nonidealities"].remove(nonideality)
            elif isinstance(nonideality["parameters"][0], int):
                nonidealities.append(
                    StuckDistribution(
                        num_of_weights=nonideality["parameters"][0],
                        G_off=simulation_parameters["G_off"],
                        G_on=simulation_parameters["G_on"],
                    )
                )
                simulation_parameters["nonidealities"].remove(nonideality)
            else:
                raise ValueError(
                    "StuckDistribution nonideality was not passed the correct parameters."
                )

        else:
            raise ValueError(f"Nonideality {nonideality} is not recognised.")

    for index, percentage in enumerate(percentages):

        # Setting percentage of faulty devices
        for nonideality in nonidealities:
            if isinstance(nonideality, StuckAtValue) or isinstance(nonideality, StuckDistribution):
                nonideality.update(percentage)

        # Running simulations
        simulation_results = _simulate(
            simulation_parameters, nonidealities, dataset, horovod=horovod
        )
        accuracies[index] = simulation_results[0]
        pre_discretisation_accuracies[index] = simulation_results[1]

    return accuracies, pre_discretisation_accuracies
