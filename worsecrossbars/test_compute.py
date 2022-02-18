import tensorflow as tf
import numpy as np

from worsecrossbars.keras_legacy.mlp_generator import mnist_mlp
from worsecrossbars.keras_legacy.nonidealities import StuckAtValue
from worsecrossbars.keras_legacy.mlp_trainer import create_datasets
from worsecrossbars.keras_legacy.mlp_trainer import train_mlp


if __name__ == "__main__":

    percentages = np.arange(0, 1.01, 0.1).round(2)
    memristor_parameters = {
        "G_off" : 7.723066346443375e-07,
        "G_on" : 2.730684400376049e-06,
        "k_V" : 0.5
    }
    mnist_dataset = create_datasets(training_validation_ratio=3)

    acc_averaged = {}


    for percentage in percentages:

        non_ideal = [
            StuckAtValue(
                memristor_parameters["G_off"],
                percentage
            )
        ]

        accuracies = []

        for model_number in range(0, 10):
            print(f"Fault % {percentage}, Model No. {model_number}")

            model = mnist_mlp(
                1,
                memristor_parameters["G_off"],
                memristor_parameters["G_on"],
                memristor_parameters["k_V"],
                non_ideal
            )

            mlp_weights, mlp_history, *_ = train_mlp(mnist_dataset, model, 1000, 100)
            
            accuracies.append(
                model.evaluate(mnist_dataset[1][0], mnist_dataset[1][1])[1]
            )
        
        acc_averaged[percentage] = sum(accuracies)/len(accuracies)
    
    print(acc_averaged)
    



