# Suppressing warnings
import os
from matplotlib.pyplot import title
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Utility imports
import numpy as np
import gc
import pickle
import argparse
import signal, sys, platform

# Source imports
from worsecrossbars.backend.MLP_generator import MNIST_MLP_1HL, MNIST_MLP_2HL, MNIST_MLP_3HL, MNIST_MLP_4HL
from worsecrossbars.backend.MLP_trainer import dataset_creation, train_MLP
from worsecrossbars.backend.fault_simulation import run_simulation
from worsecrossbars.utilities.Logging import Logging
from worsecrossbars.utilities.DropboxUpload import DropboxUpload
from worsecrossbars.utilities.MSTeamsNotifier import MSTeamsNotifier
from worsecrossbars.utilities import io_operations, initial_setup
from worsecrossbars import configs

def handler_stop_signals(signum, frame):

    if args.log is not None:
        log.write(f"Simulation terminated unexpectedly. Got signal {signal.Signals(signum).name}.\nEnding.\n")
        log.write(special="abruptend")
    if args.teams:
        webhook_url = io_operations.read_webhook()
        teams = MSTeamsNotifier(webhook_url)
        hidden_layer = "hidden layer" if args.number_hidden_layers == 1 else "hidden layers"
        teams.send_message(f"Simulation ({args.number_hidden_layers} {hidden_layer}, fault type {args.fault_type}) terminated unexpectedly.\nGot signal {signal.Signals(signum).name}.\nEnding.", title="Simulation ended", color="b90e0a")

    gc.collect()
    sys.exit(0)


def main():

    if args.setup:
        initial_setup.main_setup()
        sys.exit(0)


    # Creating the folders required to save and load the data produced by the script
    io_operations.user_folders()
    output_folder = io_operations.create_output_structure(args)

    if args.log:
        global log
        log = Logging(args.number_hidden_layers, args.fault_type, args.number_ANNs, args.number_simulations)
        log.write(special="begin")
    
    # Check configs for Dropbox / MS Teams Integration
    if args.dropbox:
        dbx = DropboxUpload(output_folder)
    if args.teams:
        webhook_url = io_operations.read_webhook()
        teams = MSTeamsNotifier(webhook_url)
        hidden_layer = "hidden layer" if args.number_hidden_layers == 1 else "hidden layers"
        teams.send_message(f"Using parameters: {args.number_hidden_layers} {hidden_layer}, fault type {args.fault_type}.", title="Started new simulation", color="028a0f")

    # Training variables setup
    MNIST_dataset = dataset_creation()
    weights_list = []
    histories_list = []
    generator_functions = {1: MNIST_MLP_1HL, 2: MNIST_MLP_2HL, 3: MNIST_MLP_3HL, 4: MNIST_MLP_4HL}

    # Model definition and training, repeated "args.number_ANNs" times to average out stochastic variancies
    for model_number in range(0, int(args.number_ANNs)):

        MNIST_MLP = generator_functions[args.number_hidden_layers](noise=args.noise, noise_variance=args.noise_variance)
        MLP_weights, MLP_history, *_ = train_MLP(MNIST_dataset, MNIST_MLP, epochs=10, batch_size=100)
        weights_list.append(MLP_weights)
        histories_list.append(MLP_history)
        gc.collect()

        if args.log:
            log.write(string=f"Trained model {model_number+1} of {args.number_ANNs}")

    # Computing training and validation loss and accuracy by averaging over all the models trained in the previous step
    if args.log:
        log.write(string=f"Done training. Computing loss and accuracy.")

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
    pickle.dump((accuracy_values, validation_accuracy_values, loss_values, validation_loss_values), \
           open(str(configs.working_dir.joinpath("outputs", output_folder, "training_validation", f"training_validation_faultType{args.fault_type}_{args.number_hidden_layers}HL_{args.noise}N_{args.noise_variance}NV.pickle")), "wb"))

    if args.log:
        log.write(string=f"Saved training and validation data.")

    # Running "args.number_simulations" simulations for each of the "args.number_ANNs" networks trained above over the specified
    # range of faulty devices percentages
    MNIST_MLP = generator_functions[args.number_hidden_layers](noise=args.noise, noise_variance=args.noise_variance)
    MNIST_MLP.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    percentages = np.arange(0, 1.01, 0.01)
    accuracies_array = np.zeros((len(weights_list), len(percentages)))

    for count, weights in enumerate(weights_list):

        accuracies_array[count] = run_simulation(percentages, weights, int(args.number_simulations), MNIST_MLP, MNIST_dataset, args.fault_type, configs.HRS_LRS_ratio, configs.number_of_conductance_levels, configs.excluded_weights_proportion)
        gc.collect()
        if args.log:
            log.write(string=f"Simulated model {count+1} of {args.number_ANNs}.")

    # Averaging the results obtained for each of the 30 sets of weights
    accuracies = np.mean(accuracies_array, axis=0, dtype=np.float64)

    # Saving accuracies array to file
    pickle.dump((percentages, accuracies, args.fault_type), \
           open(str(configs.working_dir.joinpath("outputs", output_folder, "accuracies", f"accuracies_faultType{args.fault_type}_{args.number_hidden_layers}HL_{args.noise}N_{args.noise_variance}NV.pickle")), "wb"))

    if args.log:
        log.write(special="end")
        log.close()
    
    if args.teams:
        teams.send_message(f"Finished script using parameters {args.number_hidden_layers} HL, {args.fault_type} fault type.", "Finished execution", color="028a0f")
    if args.dropbox:
        dbx.upload()


if __name__ == "__main__":

    # Command line parser for input arguments
    parser=argparse.ArgumentParser()

    parser.add_argument("-hl", dest="number_hidden_layers", metavar="HIDDEN_LAYERS", help="Number of hidden layers in the ANN", type=int, default=1, choices=range(1, 5))
    parser.add_argument("-ft", dest="fault_type", metavar="FAULT_TYPE", help="Identifier of the fault type. 1: Cannot electroform, 2: Stuck at HRS, 3: Stuck at LRS", type=int, default=1, choices=range(1, 4))
    parser.add_argument("-n", dest="noise", metavar="NOISE", help="Addition of noise to stimuli during training", type=bool, default=False)
    parser.add_argument("-nv", dest="noise_variance", metavar="NOISE_VARIANCE", help="AWGN variance for the model", type=float, default=1.0)
    parser.add_argument("-a", dest="number_ANNs", metavar="ANNS", help="Number of ANNs being simulated", type=int, default=30)
    parser.add_argument("-s", dest="number_simulations", metavar="SIMULATIONS", help="Number of simulations being run",type=int, default=30)
    parser.add_argument("-l", dest="log", metavar="LOG", help="Enable logging the output in a separate file", type=bool, default=True)
    parser.add_argument("-d", dest="dropbox", metavar="DROPBOX", help="Enable Dropbox integration", type=bool, default=True)
    parser.add_argument("-t", dest="teams", metavar="MSTEAMS", help="Enable MS Teams integration", type=bool, default=True)
    parser.add_argument("-w", dest="wipe_current", metavar="WIPE_CURRENT", type=bool, default=False)
    parser.add_argument("--setup", dest="setup",metavar="INITIAL_SETUP", type=bool, default=False)

    args=parser.parse_args()

    signal.signal(signal.SIGINT, handler_stop_signals)
    signal.signal(signal.SIGTERM, handler_stop_signals)
    if platform.system() == "Darwin" or platform.system() == "Linux":
        signal.signal(signal.SIGHUP, handler_stop_signals)

    main()
