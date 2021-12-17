"""mlp_generator_pytorch:
A backend module used to create a PyTorch model for a densely connected MLP with a given topology.
"""
from collections import OrderedDict
from typing import List
from typing import Tuple
from pathlib import Path
import warnings
import numpy as np
from sklearn.utils import shuffle
import torch
from torch import nn
from torch import Generator
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import RMSprop
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Normalize


class MNIST_MLP(nn.Module):
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
      hidden_layer_sizes: NumPy ndarray of size number_hidden_layers, contains the number of neurons
        to be instantiated in each densely-connected layer.
      noise_variance: Positive integer/float, variance of the GaussianNoise layers instantiated
        during training to boost network performance.
    """

    def __init__(self, number_hidden_layers, hidden_layer_sizes=None):

        super().__init__()

        default_neurons = {
            1: np.array([112]),
            2: np.array([100, 100]),
            3: np.array([90, 95, 95]),
            4: np.array([85, 85, 85, 85]),
        }

        # Setting default argument values
        if hidden_layer_sizes is None:
            hidden_layer_sizes = default_neurons[number_hidden_layers]

        if (
            not isinstance(hidden_layer_sizes, np.ndarray)
            or hidden_layer_sizes.size != number_hidden_layers
        ):
            raise ValueError(
                '"hidden_layer_sizes" argument should be a NumPy ndarray object '
                + "with the same size as the number of layers being instantiated."
            )

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

        # Flattening input image
        x = x.view(-1, 784)

        for layer in self.hidden:
            x = self.sigmoid(layer(x))

        output = self.softmax(self.output(x))

        return output

    @staticmethod
    def get_datasets(**kwargs) -> Tuple[Dataset, Dataset, Dataset]:
        """..."""

        # Unpacking keyword arguments
        seed = kwargs.get("seed", None)
        validation_size = kwargs.get("validation_size", 0.25)
        data_directory = kwargs.get(
            "data_directory", str(Path.home().joinpath("worsecrossbars", "utils"))
        )

        # Validating arguments
        if isinstance(validation_size, int):
            validation_size = float(validation_size)
        if not isinstance(validation_size, float) or validation_size < 0 or validation_size > 1:
            raise ValueError(
                '"validation_size" argument should be a real number comprised between ' + "0 and 1."
            )

        # Defining transforms, including standard MNIST normalisation
        normalize = Normalize((0.1307,), (0.3081,))
        transform = Compose([ToTensor(), normalize])

        # Loading the datasets
        full_training_dataset = datasets.MNIST(
            root=data_directory, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_directory, train=False, download=True, transform=transform
        )

        # Splitting the dataset
        size_full_dataset = len(full_training_dataset)
        size_validation = int(np.floor(validation_size * size_full_dataset))
        size_training = size_full_dataset - size_validation

        if seed is not None:
            training_dataset, validation_dataset = random_split(
                full_training_dataset,
                [size_training, size_validation],
                generator=Generator().manual_seed(seed),
            )
        else:
            training_dataset, validation_dataset = random_split(
                full_training_dataset, [size_training, size_validation]
            )

        temp = ([], [])

        for item in training_dataset:
            temp[0].append(item[0])
            temp[1].append(item[1])

        training_data = (torch.Tensor(temp[0]), torch.Tensor(temp[1]))

        for item in validation_dataset:
            temp[0].append(item[0])
            temp[1].append(item[1])

        for item in test_dataset:
            temp[0].append(item[0])
            temp[1].append(item[1])

        training_data[0] = torch.Tensor(training_data[0])
        training_data[1] = torch.Tensor(training_data[1])
        validation_data[0] = torch.Tensor(validation_data[0])
        validation_data[1] = torch.Tensor(validation_data[1])
        test_data[0] = torch.Tensor(test_data[0])
        test_data[1] = torch.Tensor(test_data[1])

        return training_data, validation_data, test_data

    @staticmethod
    def get_dataloader(dataset, **kwargs):
        """..."""

        # Unpacking keyword arguments
        batch_size = kwargs.get("batch_size", 100)
        shuffle = kwargs.get("shuffle", True)

        # If using CUDA, num_workers should be set to 1 and pin_memory to True.
        num_workers = kwargs.get("num_workers", 1)
        pin_memory = kwargs.get("pin_memory", True)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataloader

    def compile(self, optimizer: str, loss: str, metrics: List[str]):

        if not isinstance(optimizer, str) or optimizer.lower() not in ["rmsprop", "adam", "sgd"]:
            raise ValueError(
                '"optimiser_class" argument should be either "rmsprop", "adam" ' + 'or "sgd".'
            )

        if loss != "categorical_crossentropy":
            raise ValueError("Only categorical_crossentropy loss has been implemented so far.")

        if metrics != ["accuracy"]:
            raise ValueError("Only accuracy metric has been implemented so far.")

        # Set up optimiser
        if optimizer.lower() == "adam":
            self.optimizer = Adam(self.parameters())
        elif optimizer.lower() == "sgd":
            self.optimizer = SGD(self.parameters())
        else:
            self.optimizer = RMSprop(self.parameters())

        # Set up loss
        self.loss = CrossEntropyLoss()

        # Set up metrics
        self.metrics = metrics

    def fit(
        self,
        training_data: Dataset,
        validation_data: Dataset,
        epochs: int = 10,
        batch_size: int = 100,
        **kwargs
    ):

        # Unpacking keyword arguments
        seed = kwargs.get("seed", None)
        shuffle = kwargs.get("shuffle", True)

        # Validating arguments
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError('"epochs" argument should be an integer greater than 1.')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('"batch_size" argument should be an integer greater than 1.')
        if not isinstance(seed, int) and seed is not None:
            warnings.warn('"seed" argument should be an integer. No seed set.')
            seed = None
        if not isinstance(shuffle, bool):
            warnings.warn('"shuffle" argument should be a boolean. Defaulting to True.')
            shuffle = True

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If reproducibility is desired, a seed must be set, and cuDNN must be rendered deterministic
        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.manual_seed(seed)

        self.to(device)

        # Create training dataloader
        training_loader = MNIST_MLP.get_dataloader(
            training_data, batch_size=batch_size, shuffle=shuffle
        )

        # Run training loop
        history = []
        self.train()

        for _ in range(1, epochs + 1):

            log = OrderedDict()
            epoch_loss = 0.0

            # Run batches
            for batch_index, (data, labels) in enumerate(training_loader):

                data, labels = Variable(data).to(device), Variable(labels).to(device)

                # Backpropagation
                self.optimizer.zero_grad()
                output = self(data)
                batch_loss = self.loss(output, labels)

                batch_loss.backward()
                self.optimizer.step()

                # Update status
                epoch_loss += batch_loss.item()
                log["training_loss"] = float(epoch_loss) / (batch_index + 1)

            # Calculate training accuracy
            if self.metrics:
                y_train_pred = self.predict(training_data[0], batch_size=batch_size)

                for metric in self.metrics:
                    met_statement = metric(training_loader[1], y_train_pred)
                    log["training_" + metric.__name__] = met_statement

            # Calculate validation metrics
            y_val_pred = self.predict(validation_data[0], batch_size=batch_size)
            val_loss = self.loss(Variable(y_val_pred), Variable(validation_data[1]))
            log["validation_loss"] = val_loss.item()

            if self.metrics:
                for metric in self.metrics:
                    met_statement = metric(validation_data[1], y_val_pred)
                    log["validation_" + metric.__name__] = met_statement

            history.append(log)

        return history

    def predict(self, x, batch_size: int = 100):

        y = torch.Tensor(x.size()[0])
        data = MNIST_MLP.get_dataloader((x, y), batch_size=batch_size, shuffle=False)

        # Batch prediction
        self.eval()
        r, n = 0, x.size()[0]

        for batch_data in data:

            # Predict on batch
            x_batch = Variable(batch_data[0])
            y_batch_pred = self(x_batch).data

            # Infer prediction shape
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])

            # Add to prediction tensor
            y_pred[r : min(n, r + len(batch_data))] = y_batch_pred
            r += len(batch_data)

        return y_pred

    def evaluate(self):

        pass

    def get_weights(self):

        model_weights = []

        for layer_weights in self.parameters():
            model_weights.append(layer_weights)

        return model_weights
