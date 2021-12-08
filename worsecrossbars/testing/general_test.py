"""A general testing module:
Used to run a simple test case to verify automation
To be deleted once automation is set up"""
# Importing needed packages

import copy
import sys
from typing import List
from typing import Tuple

import numpy as np
from numpy import ndarray

import worsecrossbars.backend.weight_mapping as wm
from worsecrossbars.backend.mlp_generator import mnist_mlp
from worsecrossbars.backend.mlp_trainer import create_datasets
from worsecrossbars.backend.mlp_trainer import train_mlp

# Function to test the weight discretisation function from the weight_mapping module


def test_discretisation(network_weights: List[ndarray], simulation_parameters: dict) -> None:

    test = True
    discretised = wm.discretise_weights(network_weights, simulation_parameters)


# some sort of loop that compares discretised to an expected value?

# Function to test weight alteration function from the weight_mapping module


def test_alteration(
    network_weights: List[ndarray], failure_percentage: float, simulation_parameters: dict
) -> None:

    test = True
    altered = wm.alter_weights(network_weights, failure_percentage, simulation_parameters)

# some sort of loop to compare the altered to an expected value?
# General notes to self below, ignore them for now
# from typing import List
# "Calling the function upon which we want to conduct the tests"
# "Implementing a general test case"
# "Any code needed for the CI/Jenkins/Workflow on Github"
