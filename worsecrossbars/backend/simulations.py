"""simulations:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""

from worsecrossbars.backend.nonidealities import *
from worsecrossbars.backend.memristive_mlp import MemristiveMLP
from worsecrossbars.backend.dataloaders import mnist_dataloaders

if __name__ == "__main__":

    dataloaders = mnist_dataloaders()

    G_off = 0.0009971787221729755
    G_on = 0.003513530595228076
    k_V = 0.5

    nonideality = StuckAtValue(0, 0.25, "STUCKZERO")

    # 50% stuck at zero yields 66% accuracy over 10 runs.
    # 25% stuck at zero yields 88% accuracy over 10 runs.

    accuracies = 0

    for _ in range(10):

        model = MemristiveMLP(2, G_off=G_off, G_on=G_on, k_V=k_V, nonidealities=[nonideality])

        model.compile("rmsprop")
        weights, training_losses, validation_losses, test_loss, test_accuracy = model.fit(
            dataloaders, 10
        )

        accuracies += test_accuracy

    print(accuracies / 10)

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
