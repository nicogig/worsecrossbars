"""lightning_mlp:
A backend module used to create a PyTorch model for a densely connected MLP with a given topology.
"""
import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import Adam

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

from worsecrossbars.backend.layers import GaussianNoise
from worsecrossbars.backend.dataloaders import mnist_dataloaders


class LinearMLP(LightningModule):
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

        default_neurons = {
            1: [112],
            2: [100, 100],
            3: [90, 95, 95],
            4: [85, 85, 85, 85],
        }

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

    def training_step(self, batch, batch_idx):

        data, label = batch
        output = self(data)
        loss = cross_entropy(output, label, reduction="sum")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        data, label = batch
        output = self(data)
        loss = cross_entropy(output, label, reduction="sum")
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):

        data, label = batch
        output = self(data)
        loss = cross_entropy(output, label, reduction="sum")
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":

    train_data, val_data, test_data = mnist_dataloaders(num_workers=10, pin_memory=False)

    model = LinearMLP(2)

    # Training on CPU, for GPU pass gpus=num_gpus to the Trainer class
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    result = trainer.test(dataloaders=test_data)
    print(result)

    # Add logging?
    # Add argument parsing?
