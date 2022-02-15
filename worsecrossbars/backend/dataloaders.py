"""dataloaders:
A backend module used to create PyTorch dataloaders to perform training, validation and testing.
"""
from pathlib import Path

import numpy as np
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor


def mnist_dataloaders(**kwargs):
    """"""

    # Unpacking keyword arguments
    batch_size = kwargs.get("batch_size", 100)
    seed = kwargs.get("seed", None)
    validation_size = kwargs.get("validation_size", 0.25)
    data_directory = kwargs.get(
        "data_directory", str(Path.home().joinpath("worsecrossbars", "utils"))
    )
    shuffle = kwargs.get("shuffle", True)

    num_workers = kwargs.get("num_workers", 4)
    pin_memory = kwargs.get("pin_memory", True)

    # Validating arguments
    if isinstance(validation_size, int):
        validation_size = float(validation_size)
    if not isinstance(validation_size, float) or validation_size < 0 or validation_size > 1:
        raise ValueError(
            '"validation_size" argument should be a real number comprised between ' + "0 and 1."
        )
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('"batch_size" argument should be an integer greater than 1.')

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

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return training_loader, validation_loader, test_loader


def cifar10_dataloaders():
    """"""

    pass
