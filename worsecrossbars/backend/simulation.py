"""simulation:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from keras import Model
from numpy import ndarray

from worsecrossbars.backend.mlp_generator import mnist_mlp
from worsecrossbars.backend.mlp_trainer import train_mlp
from worsecrossbars.backend.nonidealities import D2DVariability
from worsecrossbars.backend.nonidealities import IVNonlinear
from worsecrossbars.backend.nonidealities import StuckAtValue
from worsecrossbars.backend.nonidealities import StuckDistribution
from worsecrossbars.utilities.logging_module import Logging


def _generate_linnonpres_nonidealities(
    simulation_parameters: dict,
) -> List[Union[IVNonlinear, D2DVariability]]:
    """"""

    linnonpres_nonidealities = []

    for nonideality in simulation_parameters["nonidealities"]:

        if nonideality["type"] == "IVNonlinear":
            linnonpres_nonidealities.append(
                IVNonlinear(
                    v_ref=nonideality["parameters"][0],
                    avg_gamma=nonideality["parameters"][1],
                    std_gamma=nonideality["parameters"][2],
                )
            )
            simulation_parameters["nonidealities"].remove(nonideality)

        elif nonideality["type"] == "D2DVariability":
            linnonpres_nonidealities.append(
                D2DVariability(
                    simulation_parameters["gOff"],
                    simulation_parameters["gOn"],
                    nonideality["parameters"][0],
                    nonideality["parameters"][1],
                )
            )
            simulation_parameters["nonidealities"].remove(nonideality)

    return linnonpres_nonidealities


def _generate_linpres_nonidealities(
    simulation_parameters: dict,
) -> List[Union[StuckAtValue, StuckDistribution]]:

    linpres_nonidealities = []

    for nonideality in simulation_parameters["nonidealities"]:

        if nonideality["type"] == "StuckAtValue":
            linpres_nonidealities.append(StuckAtValue(value=nonideality["parameters"][0]))
            simulation_parameters["nonidealities"].remove(nonideality)

        elif nonideality["type"] == "StuckDistribution":
            # If only nonideality["parameters"][0] is a single integer, then this indicates the
            # number of weights to create between g_off and g_on. Otherwise, if
            # nonideality["parameters"][0] is a list, this is passed as distrib to
            # StuckDistribution()
            if isinstance(nonideality["parameters"][0], list):
                linpres_nonidealities.append(
                    StuckDistribution(distrib=nonideality["parameters"][0])
                )
                simulation_parameters["nonidealities"].remove(nonideality)
            elif isinstance(nonideality["parameters"][0], int):
                linpres_nonidealities.append(
                    StuckDistribution(
                        num_of_weights=nonideality["parameters"][0],
                        g_off=simulation_parameters["gOff"],
                        g_on=simulation_parameters["gOn"],
                    )
                )
                simulation_parameters["nonidealities"].remove(nonideality)
            else:
                raise ValueError(
                    "StuckDistribution nonideality was not passed the correct parameters."
                )

        else:
            raise ValueError(f"Nonideality {nonideality} is not recognised.")

    return linpres_nonidealities


def _train_model(
    simulation_parameters: dict,
    nonidealities: list,
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    batch_size: int = 100,
    horovod: bool = False,
) -> Tuple[Model, np.ndarray]:

    model = mnist_mlp(
        simulation_parameters["gOff"],
        simulation_parameters["gOn"],
        simulation_parameters["kV"],
        nonidealities=nonidealities,
        number_hidden_layers=simulation_parameters["numberHiddenLayers"],
        noise_variance=simulation_parameters["noiseVariance"],
        horovod=horovod,
        conductance_drifting=simulation_parameters["conductanceDrifting"],
        model_size=simulation_parameters["modelSize"],
        optimiser=simulation_parameters["optimiser"],
        double_weights=simulation_parameters["doubleWeights"],
    )

    if simulation_parameters["discretisation"]:
        kwargs = {
            "discretise": simulation_parameters["discretisation"],
            "number_conductance_levels": simulation_parameters["numberConductanceLevels"],
            "excluded_weights_proportion": simulation_parameters["excludedWeightsProportion"],
            "nonidealities": nonidealities,
        }
    else:
        kwargs = {
            "nonidealities": nonidealities,
        }

    try:
        epochs = simulation_parameters["epochs"]
    except KeyError:
        epochs = 60

    *_, pre_discretisation_accuracy = train_mlp(
        dataset,
        model,
        epochs=epochs,
        batch_size=batch_size,
        hrs_lrs_ratio=simulation_parameters["gOn"] / simulation_parameters["gOff"],
        horovod=horovod,
        **kwargs,
    )

    return model, pre_discretisation_accuracy


def _simulate(
    simulation_parameters: dict,
    nonidealities: list,
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    batch_size: int = 100,
    horovod: bool = False,
    pre_trained: Model = None,
    _logger: Logging = None,
) -> Tuple[float, float]:
    """"""

    simulation_accuracies = np.zeros(simulation_parameters["numberSimulations"])
    pre_discretisation_simulation_accuracies = np.zeros(simulation_parameters["numberSimulations"])

    for simulation in range(simulation_parameters["numberSimulations"]):

        print(f"Simulation #{simulation+1}, nonidealities: {nonidealities}")

        if horovod:
            import horovod.tensorflow as hvd

            if hvd.rank() == 0:
                _logger.write(
                    f"Performing simulation {simulation+1}. Nonidealities {nonidealities}"
                )
        else:
            _logger.write(f"Performing simulation {simulation+1}. Nonidealities {nonidealities}")

        if pre_trained:
            # Assigning ideal model and accuracies
            new_model = mnist_mlp(
                simulation_parameters["gOff"],
                simulation_parameters["gOn"],
                simulation_parameters["kV"],
                nonidealities=nonidealities,
                number_hidden_layers=simulation_parameters["numberHiddenLayers"],
                noise_variance=simulation_parameters["noiseVariance"],
                horovod=horovod,
                conductance_drifting=simulation_parameters["conductanceDrifting"],
                model_size=simulation_parameters["modelSize"],
                optimiser=simulation_parameters["optimiser"],
                double_weights=simulation_parameters["doubleWeights"],
            )
            new_model.build((1, 784))

            for index, layer in enumerate(new_model.layers):
                layer.set_weights(pre_trained.layers[index].get_weights())

            pre_discretisation_accuracy = new_model.evaluate(dataset[1][0], dataset[1][1])[1]
        else:
            # Training model with given nonidealities
            model, pre_discretisation_accuracy = _train_model(
                simulation_parameters=simulation_parameters,
                nonidealities=nonidealities,
                dataset=dataset,
                batch_size=batch_size,
                horovod=horovod,
            )

        if horovod and hvd.rank() == 0:
            _logger.write(f"Finished. Accuracy {pre_discretisation_accuracy}")
        else:
            _logger.write(f"Finished. Accuracy {pre_discretisation_accuracy}")

        if simulation_parameters["discretisation"]:
            simulation_accuracies[simulation] = model.evaluate(dataset[1][0], dataset[1][1])[1]
            pre_discretisation_simulation_accuracies[simulation] = pre_discretisation_accuracy

        else:
            simulation_accuracies[simulation] = pre_discretisation_accuracy

    # Returning 0.0 for average_pre_discretisation_accuracy if no discretisation is being performed
    average_accuracy = simulation_accuracies.mean()
    average_pre_discretisation_accuracy = pre_discretisation_simulation_accuracies.mean()

    return average_accuracy, average_pre_discretisation_accuracy


def run_simulations(
    simulation_parameters: dict,
    dataset: Tuple[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]],
    batch_size: int = 100,
    horovod: bool = False,
    logger: Logging = None,
) -> Tuple[ndarray, ndarray]:
    """This function...

    Returns:
      accuracies: Numpy ndarray containing one (or multiple) float value(s),
      pre_discretisation_accuracies: Numpy ndarray containing one (or multiple) float value(s),
    """

    # Generating linearity non-preserving nonidealities
    nonidealities = _generate_linnonpres_nonidealities(simulation_parameters)

    if not simulation_parameters["nonidealities"]:

        # If no other nonidealities (i.e. no device-percentage-based nonidealities remain), there
        # is no need to simualate varying percentages of faulty devices.
        simulation_results = _simulate(
            simulation_parameters,
            nonidealities,
            dataset,
            batch_size=batch_size,
            horovod=horovod,
            _logger=logger,
        )
        accuracies = np.array([simulation_results[0]])
        pre_discretisation_accuracies = np.array([simulation_results[1]])

        return accuracies, pre_discretisation_accuracies

    # Generating vectors to be used for device-percentage-based nonidealities analysis
    percentages = np.arange(0.0, 1.01, 0.05).round(2)
    accuracies = np.zeros(percentages.size)
    pre_discretisation_accuracies = np.zeros(percentages.size)

    # Adding percentage-based nonidealities
    nonidealities.extend(_generate_linpres_nonidealities(simulation_parameters))

    # Handling nonidealities_after_training frameworks
    try:
        nonidealities_after_training = simulation_parameters["nonidealitiesAfterTraining"]
    except KeyError:
        nonidealities_after_training = 0

    trained_models = []

    if nonidealities_after_training:

        # Training ideal memristive models
        for model in range(nonidealities_after_training):
            print(f"Training NAT model {model+1}.")

            if horovod:
                import horovod.tensorflow as hvd

                if hvd.rank() == 0:
                    logger.write(f"Training NAT model {model+1}.")
            else:
                logger.write(f"Training NAT model {model+1}.")

            trained_models.append(
                _train_model(
                    simulation_parameters=simulation_parameters,
                    nonidealities=[],
                    dataset=dataset,
                    batch_size=batch_size,
                    horovod=horovod,
                )[0]
            )

    for index, percentage in enumerate(percentages):

        # Setting percentage of faulty devices
        for nonideality in nonidealities:
            if isinstance(nonideality, StuckAtValue) or isinstance(nonideality, StuckDistribution):
                _ = nonideality.update(percentage)

        if trained_models:
            # Nonidealities after training simulations
            for model in trained_models:
                simulation_results = _simulate(
                    simulation_parameters,
                    nonidealities,
                    dataset,
                    horovod=horovod,
                    pre_trained=model,
                    _logger=logger,
                )

                accuracies[index] += simulation_results[0]
                pre_discretisation_accuracies[index] += simulation_results[1]

        else:
            # Regular simulations
            simulation_results = _simulate(
                simulation_parameters,
                nonidealities,
                dataset,
                horovod=horovod,
                _logger=logger,
            )
            accuracies[index] = simulation_results[0]
            pre_discretisation_accuracies[index] = simulation_results[1]

    if nonidealities_after_training:
        accuracies /= nonidealities_after_training
        pre_discretisation_accuracies /= nonidealities_after_training

    return accuracies, pre_discretisation_accuracies
