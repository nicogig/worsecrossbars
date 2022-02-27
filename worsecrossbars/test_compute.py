import numpy as np

from worsecrossbars.backend.mlp_generator import mnist_mlp
from worsecrossbars.backend.nonidealities import StuckAtValue
from worsecrossbars.backend.mlp_trainer import create_datasets
from worsecrossbars.backend.mlp_trainer import train_mlp


if __name__ == "__main__":

    # Defining simulation parameters
    percentages = np.arange(0, 1.01, 0.1).round(2)
    number_simulations = 10
    epochs = 2
    batch_size = 100

    number_hidden_layers = 2
    noise_variance = 0.0
    fault_type = "StuckOff"

    # Hard-coding memristive parameters
    memristive_parameters = {
        "G_off": 7.723066346443375e-07,
        "G_on": 2.730684400376049e-06,
        "k_V": 0.5,
    }
    hrs_lrs_ratio = memristive_parameters["G_on"] / memristive_parameters["G_off"]
    number_conductance_levels = 10
    excluded_weights_proportion = 0.015

    # Generating MNIST dataset
    mnist_dataset = create_datasets(training_validation_ratio=3)

    # Generating empty vector to store accuracies
    accuracies = np.zeros(percentages.size)

    for index, percentage in enumerate(percentages):

        nonidealities = [StuckAtValue(memristive_parameters["G_off"], percentage)]

        simulation_accuracies = np.zeros(number_simulations)

        for simulation in range(number_simulations):

            print(f"{percentage}% faulty devices, simulation {simulation+1}")

            model = mnist_mlp(
                memristive_parameters["G_off"],
                memristive_parameters["G_on"],
                memristive_parameters["k_V"],
                nonidealities,
                number_hidden_layers=number_hidden_layers,
                noise_variance=noise_variance,
            )

            mlp_weights, mlp_history = train_mlp(
                mnist_dataset,
                model,
                epochs,
                batch_size,
                discretise=True,
                hrs_lrs_ratio=hrs_lrs_ratio,
                number_conductance_levels=number_conductance_levels,
                excluded_weights_proportion=excluded_weights_proportion,
            )

            simulation_accuracies[simulation] = model.evaluate(
                mnist_dataset[1][0], mnist_dataset[1][1]
            )[1]

            print(simulation_accuracies)

        accuracies[index] = simulation_accuracies.mean()

    print(accuracies)
