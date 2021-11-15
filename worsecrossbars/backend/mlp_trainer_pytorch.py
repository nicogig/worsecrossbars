"""
mlp_trainer_pytorch:
A backend module used to instantiate the MNIST dataset and train a PyTorch model on it.
"""

from pathlib import Path
import warnings
import numpy as np
import torch
from torch import cuda
from torch import Generator
from torch import manual_seed
from torch.optim import RMSprop, Adam, SGD
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Normalize

from worsecrossbars.backend.mlp_generator_pytorch import MNIST_MLP
import time


def get_data_loaders(**kwargs):
    """
    # If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """

    # Unpacking keyword arguments
    batch_size = kwargs.get("batch_size", 100)
    seed = kwargs.get("seed", None)
    validation_size = kwargs.get("validation_size", 0.25)
    data_directory = kwargs.get("data_directory",
                                str(Path.home().joinpath("worsecrossbars", "utils")))
    shuffle = kwargs.get("shuffle", True)
    num_workers = kwargs.get("num_workers", 1)
    pin_memory = kwargs.get("pin_memory", True)

    # Validating arguments
    if isinstance(validation_size, int):
        validation_size = float(validation_size)
    if not isinstance(validation_size, float) or validation_size < 0 or validation_size > 1:
        raise ValueError("\"validation_size\" argument should be a real number comprised between " +
                         "0 and 1.")

    # Defining transforms, including standard MNIST normalisation
    normalize = Normalize((0.1307,), (0.3081,))
    transform = Compose([ToTensor(),normalize])

    # Loading the datasets
    full_training_dataset = datasets.MNIST(root=data_directory, train=True, download=True,
                                           transform=transform)
    test_dataset = datasets.MNIST(root=data_directory, train=False, download=True,
                                  transform=transform)

    # Splitting the dataset 
    size_full_dataset = len(full_training_dataset)
    size_validation = int(np.floor(validation_size * size_full_dataset))
    size_training = size_full_dataset - size_validation

    if seed is not None:
        training_dataset, validation_dataset = random_split(full_training_dataset,
                                                            [size_training, size_validation],
                                                            generator=Generator().manual_seed(seed))
    else:
        training_dataset, validation_dataset = random_split(full_training_dataset,
                                                            [size_training, size_validation])       

    training_loader = DataLoader(training_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers,
                                 pin_memory=pin_memory)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers,
                                   pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory)

    return training_loader, validation_loader, test_loader


def train_pytorch(model, epochs, **kwargs):
    """
    """

    # Unpacking keyword arguments
    seed = kwargs.get("seed", None)
    batch_size = kwargs.get("batch_size", 100)
    optimiser_class = kwargs.get("optimiser_class", "rmsprop")

    # Validating arguments
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("\"epochs\" argument should be an integer greater than 1.")
    if not isinstance(seed, int) and seed is not None:
        warnings.warn("\"seed\" argument should be an integer. No seed set.")
        seed = None
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("\"batch_size\" argument should be an integer greater than 1.")
    if not isinstance(optimiser_class, str) or \
        optimiser_class.lower() not in ["rmsprop", "adam", "sgd"]:
        raise ValueError("\"optimiser_class\" argument should be either \"rmsprop\", \"adam\" " +
                         "or \"sgd\".")

    training_loader, validation_loader, test_loader = get_data_loaders(batch_size=batch_size,
                                                                       seed=seed)

    # Results lists
    training_losses = []
    validation_losses = []
    test_loss = None
    test_accuracy = None

    # Device configuration
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    # If reproducibility is desired, a seed must be set, and cuDNN must be disabled, as it uses
    # nondeterministic algorithms
    if seed is not None:
        torch.backends.cudnn.enabled = False
        manual_seed(seed)

    # Sending network model to GPU if available
    if cuda.is_available():
        model = model.cuda()

    # Set up optimiser
    if optimiser_class.lower() == "adam":
        optimiser = Adam(model.parameters())
    elif optimiser_class.lower() == "sgd":
        optimiser = SGD(model.parameters())
    else:
        optimiser = RMSprop(model.parameters())

    # Training, validation and testing
    for epoch in range(1, epochs + 1):

        # Training step
        training_loss = 0.0
        model.train()

        for data, label in training_loader:

            if cuda.is_available():
                data, label = data.cuda(), label.cuda()

            optimiser.zero_grad()

            output = model(data)
            loss = cross_entropy(output, label, reduction="sum")

            loss.backward()
            optimiser.step()

            #.item() method should be removed from the following line if tensors are preferred
            # over floats
            training_loss += loss.item()

        # Validation step
        validation_loss = 0.0
        model.eval()

        with torch.no_grad():

            for data, label in validation_loader:

                if cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                
                output = model(data)

                #.item() method should be removed from the following line if tensors are preferred
                # over floats
                validation_loss += cross_entropy(output, label, reduction="sum").item()
        
        training_losses.append(training_loss/len(training_loader.dataset))
        validation_losses.append(validation_loss/len(validation_loader.dataset))

    # Testing        
    test_loss = 0.0
    correct = 0.0
    model.eval()

    with torch.no_grad():

        for data, label in test_loader:

            if cuda.is_available():
                data, label = data.cuda(), label.cuda()

            output = model(data)

            #.item() method should be removed from the following line if tensors are preferred
            # over floats
            test_loss += cross_entropy(output, label, reduction="sum").item()
            prediction = output.data.max(1, keepdim=True)[1]

            #.item() method should be removed from the following line if tensors are preferred
            # over floats
            correct += prediction.eq(label.data.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    model_weights = []
    for layer_weights in model.parameters():
        model_weights.append(layer_weights)

    return model_weights, training_losses, validation_losses, test_loss, test_accuracy


if __name__ == "__main__":

    start = time.time()

    model = MNIST_MLP(2)

    model_weights, training_losses, validation_losses, test_loss, test_accuracy = train_pytorch(model, 10)

    end = time.time()
    print(f"Time elapsed: {end - start}")

    # print(model_weights)
    # print(training_losses)
    # print(validation_losses)
    # print(test_loss)
    # print(test_accuracy)
