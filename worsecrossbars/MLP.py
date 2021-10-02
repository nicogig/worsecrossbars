# Suppressing warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Utility imports
import numpy as np
import gc
import pickle
import argparse

# Source imports
from worsecrossbars.backend.mlp_generator import one_layer, two_layers, three_layers, four_layers
from worsecrossbars.backend.mlp_trainer import dataset_creation, train_MLP
from worsecrossbars.backend.fault_simulation import run_simulation
from worsecrossbars.utilities.spruce_logging import Logging
from worsecrossbars.utilities.upload_to_dropbox import check_auth_presence, upload
from worsecrossbars.utilities.msteams_notifier import check_webhook_presence, send_message
from worsecrossbars.utilities import create_folder_structure
from worsecrossbars import configs


def main():

    create_folder_structure.user_folders()


    if args.log:
        log = Logging(args.number_hidden_layers, args.fault_type, args.number_ANNs, args.number_simulations)
        log.write(special="begin")
    
    # Check configs for Dropbox / MS Teams Integration
    if args.dropbox:
        check_auth_presence()
    if args.teams:
        check_webhook_presence()
        send_message(f"Using parameters: {args.number_hidden_layers} HL, {args.fault_type} fault type.", title="Started new simulation", color="028a0f")

    # Training variables setup
    MNIST_dataset = dataset_creation()
    weights_list = []
    histories_list = []
    generator_functions = {1: one_layer, 2:two_layers, 3:three_layers, 4:four_layers}

    # Model definition and training, repeated "args.number_ANNs" times to average out stochastic variancies
    for model_number in range(0, int(args.number_ANNs)):

        MNIST_MLP = generator_functions[args.number_hidden_layers]()
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
    pickle.dump((accuracy_values, validation_accuracy_values, loss_values, validation_loss_values), open(str(configs.working_dir.joinpath("outputs", "training_validation", f"training_validation_faultType{args.fault_type}_{args.number_hidden_layers}HL.pickle")), "wb"))

    if args.log:
        log.write(string=f"Saved training and validation data.")

    # Running "args.number_simulations" simulations for each of the "args.number_ANNs" networks trained above over the specified
    # range of faulty devices percentages
    MNIST_MLP = generator_functions[args.number_hidden_layers]()
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
    pickle.dump((percentages, accuracies, args.fault_type), open(str(configs.working_dir.joinpath("outputs", "accuracies", f"accuracies_faultType{args.fault_type}_{args.number_hidden_layers}HL.pickle")), "wb"))

    if args.log:
        log.write(special="end")
        log.close()
    
    if args.teams:
        send_message(f"Finished script using parameters {args.number_hidden_layers} HL, {args.fault_type} fault type.", "Finished execution", color="028a0f")
    if args.dropbox:
        upload(args.fault_type, args.number_hidden_layers)



if __name__ == "__main__":

    # Command line parser for input arguments
    parser=argparse.ArgumentParser()

    parser.add_argument("-hl", dest="number_hidden_layers", metavar="HIDDEN_LAYERS", help="Number of hidden layers in the ANN", type=int, default=1, choices=range(1, 5))
    parser.add_argument("-ft", dest="fault_type", metavar="FAULT_TYPE", help="Identifier of the fault type. 1: Cannot electroform, 2: Stuck at HRS, 3: Stuck at LRS", type=int, default=1, choices=range(1, 4))
    parser.add_argument("-a", dest="number_ANNs", metavar="ANNS", help="Number of ANNs being simulated", type=int, default=30)
    parser.add_argument("-s", dest="number_simulations", metavar="SIMULATIONS", help="Number of simulations being run",type=int, default=30)
    parser.add_argument("-l", dest="log", metavar="LOG", help="Enable logging the output in a separate file", type=bool, default=False)
    parser.add_argument("-d", dest="dropbox", metavar="DROPBOX", help="Enable Dropbox integration", type=bool, default=False)
    parser.add_argument("-t", dest="teams", metavar="MSTEAMS", help="Enable MS Teams integration", type=bool, default=False)

    args=parser.parse_args()

    main()
