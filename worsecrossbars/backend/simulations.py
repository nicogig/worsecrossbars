"""simulations:
A backend module used to simulate the effect of faulty devices on memristive ANN performance.
"""

from worsecrossbars.backend.nonidealities import *
from worsecrossbars.backend.memristive_mlp import MemristiveMLP
from worsecrossbars.backend.linear_mlp import LinearMLP
from worsecrossbars.backend.dataloaders import mnist_dataloaders

if __name__ == "__main__":

    dataloaders = mnist_dataloaders()

    G_off = 0.0009971787221729755
    G_on = 0.003513530595228076
    k_V = 0.5

    nonideality = StuckAtValue(0, 0.5, "STUCKZERO")

    model1 = MemristiveMLP(2, G_off=G_off, G_on=G_on, k_V=k_V, nonidealities=[nonideality])
    model2 = LinearMLP(2, noise_sd=0.5)

    model1.compile("rmsprop")
    weights, training_losses, validation_losses, test_loss, test_accuracy = model1.fit(
        dataloaders, 10
    )

    print(training_losses)
    print(validation_losses)
    print(test_loss)
    print(test_accuracy)

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
