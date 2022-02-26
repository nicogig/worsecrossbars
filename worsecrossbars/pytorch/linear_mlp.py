"""memristive_mlp:
A backend module used to create a PyTorch model for a densely connected MLP with a given topology.
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim import RMSprop
from torch.optim import SGD
from torch.utils.data import DataLoader

from worsecrossbars.pytorch.layers import GaussianNoise


class LinearMLP(nn.Module):
    """This class implements a PyTorch model set up to be trained to recognise digits from the
    MNIST dataset (784 input neurons, 10 softmax output neurons).

    The network architecture corresponds to that employed in the "Simulation of Inference Accuracy
    Using Realistic RRAM Devices" paper. It consists of a feed-forward multilayer perceptron with
    784 input neurons (encoding pixel intensities for 28 x 28 pixel MNIST images), two 100-neuron
    hidden layers, and 10 output neurons (each corresponding to one of the ten digits). The first
    three layers employ a sigmoid activation function, whilst the output layer makes use of a
    softmax activation function (which means a cross-entropy error function is then used
    throughout the learning task). All 60,000 MNIST training images were employed, divided into
    training and validation sets in a 3:1 ratio, as described in the aforementioned paper.

    For the one-layer, three-layers and four-layers topologies, the number of neurons in each
    hidden layer was tweaked so as to produce a final network with about the same number of
    trainable parameters as the original, two-layers ANN. This was done to ensure that variability
    in fault simulation results was indeeed due to the number of layers being altered, rather than
    to a different number of weights being implemented.

    The function also gives the user the option to add GaussianNoise layers with a specific variance
    between hidden layers during training. This is done to increase the network's generalisation
    power, as well as to increase resilience to faulty memristive devices.

    Args:
      number_hidden_layers: Integer comprised between 1 and 4, number of hidden layers instantiated
        as part of the model.
      hidden_layer_sizes: List of size number_hidden_layers, contains the number of neurons to be
        instantiated in each densely-connected layer.
      noise_sd: Positive integer/float, relative standard deviation of the GaussianNoise layers
        instantiated during training to boost network performance.
    """

    def __init__(
        self,
        number_hidden_layers: int,
        hidden_layer_sizes: list = None,
        noise_sd: float = 0.0,
        device: torch.device = None,
    ) -> None:

        super().__init__()

        # Define optimiser
        self.optimiser: Optimizer = None

        # Set up loss function
        self.loss = CrossEntropyLoss(reduction="sum")

        default_neurons = {
            1: [112],
            2: [100, 100],
            3: [90, 95, 95],
            4: [85, 85, 85, 85],
        }

        # Setting up device
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setting default argument values
        if hidden_layer_sizes is None:
            hidden_layer_sizes = default_neurons[number_hidden_layers]

        if (
            not isinstance(hidden_layer_sizes, list)
            or len(hidden_layer_sizes) != number_hidden_layers
        ):
            raise ValueError(
                '"hidden_layer_sizes" argument should be a list object with the same size as the'
                + "number of layers being instantiated."
            )

        # Noise layers
        self.noise = GaussianNoise(self.device, noise_sd)

        # Activation layers
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # Hidden layers
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(784, hidden_layer_sizes[0]))

        for index in range(number_hidden_layers - 1):
            self.hidden.append(nn.Linear(hidden_layer_sizes[index], hidden_layer_sizes[index + 1]))

        # Output layer
        self.output = nn.Linear(hidden_layer_sizes[-1], 10)

    def forward(self, x):
        """"""

        # Flattening input image
        x = x.view(-1, 784)

        for layer in self.hidden:
            x = self.sigmoid(self.noise(layer(x)))

        output = self.softmax(self.output(x))

        return output

    def compile(self, optimiser: str) -> None:
        """"""

        if not isinstance(optimiser, str) or optimiser.lower() not in ["rmsprop", "adam", "sgd"]:
            raise ValueError(
                '"optimiser_class" argument should be either "rmsprop", "adam" ' + 'or "sgd".'
            )

        # Set up optimiser
        if optimiser.lower() == "adam":
            self.optimiser = Adam(self.parameters())
        elif optimiser.lower() == "sgd":
            self.optimiser = SGD(self.parameters())
        else:
            self.optimiser = RMSprop(self.parameters())

        # Send model to device
        self.to(self.device)

    def fit(self, dataloaders: Tuple[DataLoader, DataLoader, DataLoader], epochs: int):
        """This function trains the Pytorch model on the dataset provided to it.

        Args:
          dataloaders: Tuple containing training, validation and testing dataloaders, as provided by
            the appropriate function in dataloaders.py.
          epochs: Positive integer, number of epochs used in training.

        Returns:
          weights:
          training_losses:
          validation_losses:
          test_loss:
          test_accuracy:
        """

        # Results lists
        training_losses = []
        validation_losses = []

        # Training, validation and testing
        for _ in range(1, epochs + 1):

            # Training step
            training_loss = 0.0
            self.train()

            for data, label in dataloaders[0]:

                data, label = data.to(self.device), label.to(self.device)

                self.optimiser.zero_grad()

                output = self(data)
                loss = self.loss(output, label)

                loss.backward()
                self.optimiser.step()

                training_loss += loss.item()

            # Validation step
            validation_loss = 0.0
            self.eval()

            with torch.no_grad():

                for data, label in dataloaders[1]:

                    data, label = data.to(self.device), label.to(self.device)

                    output = self(data)

                    validation_loss += self.loss(output, label).item()

            training_losses.append(training_loss / len(dataloaders[0].dataset))
            validation_losses.append(validation_loss / len(dataloaders[1].dataset))

        # Testing
        test_loss = 0.0
        correct = 0.0
        self.eval()

        with torch.no_grad():

            for data, label in dataloaders[2]:

                data, label = data.to(self.device), label.to(self.device)

                output = self(data)

                test_loss += self.loss(output, label).item()
                prediction = output.data.max(1, keepdim=True)[1]

                correct += prediction.eq(label.data.view_as(prediction)).sum().item()

        test_loss /= len(dataloaders[2].dataset)
        test_accuracy = 100.0 * correct / len(dataloaders[2].dataset)

        weights = []
        for layer_weights in self.parameters():
            weights.append(layer_weights)

        return weights, training_losses, validation_losses, test_loss, test_accuracy
