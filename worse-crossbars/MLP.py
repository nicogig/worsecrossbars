# Suppressing warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Utility imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import gc
import pickle
import sys, argparse, signal
from datetime import datetime

# Source imports
from backend.mlp_generator import one_layer, two_layers, three_layers, four_layers
from backend.mlp_trainer import dataset_creation, train_MLP
from backend.fault_simulation import run_simulation

# Command line parser for input arguments

parser=argparse.ArgumentParser()

parser.add_argument("--number_hidden_layers", help="Number of hidden layers in the ANN", type=int, default=1)
parser.add_argument("--fault_type", help="Identifier of the fault type", type=int, default=1)
parser.add_argument("--number_ANNs", help="Number of ANNs being simulated", type=int, default=30)
parser.add_argument("--number_simulations", help="Number of simulations being run",type=int, default=30)
parser.add_argument("--log", help="Enable logging the output in a separate file", type=bool, default=False)

args=parser.parse_args()

def signal_term_handler(signal, frame):
    if args.log:
        with open('spruce.log', 'a') as the_log:
            the_log.write(f'[{datetime.now().strftime("%H:%M:%S")}] Got {signal}. Ending process.\n')
            the_log.write(f'----- End Log {datetime.now().__str__()} -----')
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_term_handler)
signal.signal(signal.SIGINT, signal_term_handler)

if args.log:
    with open('spruce.log', 'a') as the_log:
        the_log.write(f'\n----- Begin Log {datetime.now().__str__()} -----\n')
        the_log.write(f'Attempting Simulation with following parameters:\n')
        the_log.write(f'number_hidden_layers: {args.number_hidden_layers}\n')
        the_log.write(f'fault_type: {args.fault_type}\n')
        the_log.write(f'number_ANNs: {args.number_ANNs}\n')
        the_log.write(f'number_simulations: {args.number_simulations}\n\n')


# The statements below are employed to control the mode in which the Jupyter
# notebook operates.

# Legal values: [1, 2, 3, 4]
number_of_hidden_layers = args.number_hidden_layers

# Legal values: [1, 2, 3]; 1: Cannot electroform, 2: Stuck at HRS, 3: Stuck at LRS
fault_type = args.fault_type

# Device parameters
HRS_LRS_ratio = 5
excluded_weights_proportion = 0.015


# Create dataset and results lists
MNIST_dataset = dataset_creation()
weights_list = []
histories_list = []

# Helper values
generator_functions = {1: one_layer, 2:two_layers, 3:three_layers, 4:four_layers}
number_of_ANNs = args.number_ANNs


# Model definition and training, repeated 30 times to average out stochastic variancies
for model_number in range(0, int(number_of_ANNs)):

    MNIST_MLP = generator_functions[number_of_hidden_layers]()
    MLP_weights, MLP_history, *_ = train_MLP(MNIST_dataset, MNIST_MLP, epochs=10, batch_size=100)
    weights_list.append(MLP_weights)
    histories_list.append(MLP_history)
    gc.collect()
    if args.log:
        with open('spruce.log', 'a') as the_log:
            the_log.write(f'[{datetime.now().strftime("%H:%M:%S")}] Trained Model {model_number} of {number_of_ANNs}\n')



# Computing training and validation loss and accuracy by averaging over all the models trained in the previous step

if args.log:
    with open('spruce.log', 'a') as the_log:
        the_log.write(f'[{datetime.now().strftime("%H:%M:%S")}] Done training. Computing loss and accuracy.\n')

epochs = range(1, len(histories_list[0].history["accuracy"]) + 1)
accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
validation_accuracy_values = np.zeros(len(histories_list[0].history["accuracy"]))
loss_values = np.zeros(len(histories_list[0].history["accuracy"]))
validation_loss_values = np.zeros(len(histories_list[0].history["accuracy"]))

for MLP_history in histories_list:

    history_dict = MLP_history.history
    accuracy_values += np.array(history_dict["accuracy"])
    validation_accuracy_values += np.array(history_dict["val_accuracy"])
    loss_values += np.array(history_dict["loss"])
    validation_loss_values += np.array(history_dict["val_loss"])

accuracy_values /= len(histories_list)
validation_accuracy_values /= len(histories_list)
loss_values /= len(histories_list)
validation_loss_values /= len(histories_list)

# Saving training/validation data to file
pickle.dump((accuracy_values, validation_accuracy_values, loss_values, validation_loss_values), open("./saved_data/training_validation_{}HL.p".format(number_of_hidden_layers), "wb"))

if args.log:
    with open('spruce.log', 'a') as the_log:
        the_log.write(f'[{datetime.now().strftime("%H:%M:%S")}] Saved training and validation data.\n')

# Running "number_of_simulations" simulations for each of the "number_of_ANNs" networks trained above over the specified
# range of faulty devices percentages

MNIST_MLP = generator_functions[number_of_hidden_layers]()
MNIST_MLP.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

percentages = np.arange(0, 1.01, 0.01)
accuracies_array = np.zeros((len(weights_list), len(percentages)))

number_of_simulations = args.number_simulations

for count, weights in enumerate(weights_list):

    accuracies_array[count] = run_simulation(percentages, weights, int(number_of_simulations), MNIST_MLP, MNIST_dataset, fault_type, HRS_LRS_ratio, excluded_weights_proportion)
    gc.collect()
    if args.log:
        with open('spruce.log', 'a') as the_log:
            the_log.write(f'[{datetime.now().strftime("%H:%M:%S")}] Simulated Model {count} of {number_of_ANNs}.\n')

# Averaging the results obtained for each of the 30 sets of weights
accuracies = np.mean(accuracies_array, axis=0, dtype=np.float64)

# Saving accuracies array to file
pickle.dump((percentages, accuracies, fault_type), open("./saved_data/accuracies_faultType{}_{}HL.p".format(fault_type, number_of_hidden_layers), "wb"))

if args.log:
    with open('spruce.log', 'a') as the_log:
        the_log.write(f'[{datetime.now().strftime("%H:%M:%S")}] Saved accuracies to file. Ending gracefully.\n')
        the_log.write(f'----- End Log {datetime.now().__str__()} -----')
