"""mlp_trainer_pytorch:
A backend module used to instantiate the MNIST dataset and train a PyTorch model on it.
"""
from typing import List
from typing import Tuple
from pathlib import Path
import warnings
import numpy as np
import torch
from torch import cuda
from torch import Generator
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Normalize

from worsecrossbars.backend.mlp_generator_pytorch import MNIST_MLP
import time


# def get_data_loaders(**kwargs):

#     # Unpacking keyword arguments
#     batch_size = kwargs.get("batch_size", 100)
#     seed = kwargs.get("seed", None)
#     validation_size = kwargs.get("validation_size", 0.25)
#     data_directory = kwargs.get(
#         "data_directory", str(Path.home().joinpath("worsecrossbars", "utils"))
#     )
#     shuffle = kwargs.get("shuffle", True)

#     # If using CUDA, num_workers should be set to 1 and pin_memory to True.
#     num_workers = kwargs.get("num_workers", 1)
#     pin_memory = kwargs.get("pin_memory", True)

#     # Validating arguments
#     if isinstance(validation_size, int):
#         validation_size = float(validation_size)
#     if not isinstance(validation_size, float) or validation_size < 0 or validation_size > 1:
#         raise ValueError(
#             '"validation_size" argument should be a real number comprised between ' + "0 and 1."
#         )
#     if not isinstance(batch_size, int) or batch_size < 1:
#         raise ValueError('"batch_size" argument should be an integer greater than 1.')

#     # Defining transforms, including standard MNIST normalisation
#     normalize = Normalize((0.1307,), (0.3081,))
#     transform = Compose([ToTensor(), normalize])

#     # Loading the datasets
#     full_training_dataset = datasets.MNIST(
#         root=data_directory, train=True, download=True, transform=transform
#     )
#     test_dataset = datasets.MNIST(
#         root=data_directory, train=False, download=True, transform=transform
#     )

#     # Splitting the dataset
#     size_full_dataset = len(full_training_dataset)
#     size_validation = int(np.floor(validation_size * size_full_dataset))
#     size_training = size_full_dataset - size_validation

#     if seed is not None:
#         training_dataset, validation_dataset = random_split(
#             full_training_dataset,
#             [size_training, size_validation],
#             generator=Generator().manual_seed(seed),
#         )
#     else:
#         training_dataset, validation_dataset = random_split(
#             full_training_dataset, [size_training, size_validation]
#         )

#     training_loader = DataLoader(
#         training_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )

#     validation_loader = DataLoader(
#         validation_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )

#     return training_loader, validation_loader, test_loader


# def train_pytorch(data_loaders: Tuple[DataLoader, DataLoader, DataLoader], model, epochs, **kwargs):
#     """This function trains a given Pytorch model on the dataset provided to it.

#     Args:
#       data_loaders: Tuple containing training, validation and testing dataloaders, as
#         provided by the get_data_loaders() function defined above.
#       model: Pytorch model which is to be trained
#       epochs: Positive integer, number of epochs used in training.

#     Returns:

#     """

#     # Results lists
#     training_losses = []
#     validation_losses = []
#     test_loss = None
#     test_accuracy = None

#     # Training, validation and testing
#     for _ in range(1, epochs + 1):

#         # Training step
#         training_loss = 0.0
#         model.train()

#         for data, label in data_loaders[0]:

#             data, label = data.to(device), label.to(device)

#             optimiser.zero_grad()

#             output = model(data)
#             loss = cross_entropy(output, label, reduction="sum")

#             loss.backward()
#             optimiser.step()

#             training_loss += loss.item()

#         # Validation step
#         validation_loss = 0.0
#         model.eval()

#         with torch.no_grad():

#             for data, label in data_loaders[1]:

#                 data, label = data.to(device), label.to(device)

#                 output = model(data)

#                 validation_loss += cross_entropy(output, label, reduction="sum").item()

#         training_losses.append(training_loss / len(data_loaders[0].dataset))
#         validation_losses.append(validation_loss / len(data_loaders[1].dataset))

#     # Testing
#     test_loss = 0.0
#     correct = 0.0
#     model.eval()

#     with torch.no_grad():

#         for data, label in data_loaders[2]:

#             data, label = data.to(device), label.to(device)

#             output = model(data)

#             test_loss += cross_entropy(output, label, reduction="sum").item()
#             prediction = output.data.max(1, keepdim=True)[1]

#             correct += prediction.eq(label.data.view_as(prediction)).sum().item()

#     test_loss /= len(data_loaders[2].dataset)
#     test_accuracy = 100.0 * correct / len(data_loaders[2].dataset)

#     model_weights = []
#     for layer_weights in model.parameters():
#         model_weights.append(layer_weights)

#     return model_weights, training_losses, validation_losses, test_loss, test_accuracy


if __name__ == "__main__":

    start = time.time()

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    model = MNIST_MLP(2)

    data_loaders = MNIST_MLP.get_datasets()

    # print(data_loaders[0])

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    mlp_history = model.fit(
        data_loaders[0],
        data_loaders[1],
        epochs=10,
        batch_size=100,
    )
    print(mlp_history)

    end = time.time()
    print(f"Time elapsed: {end - start}")

    # print(model_weights)
    # print(training_losses)
    # print(validation_losses)
    # print(test_loss)
    # print(test_accuracy)
