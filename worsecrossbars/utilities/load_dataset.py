from pathlib import Path
from typing import Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor


def load_dataset(
    device: torch.device, test_batch_size: int, train_batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    home_dir = Path.home().joinpath("worsecrossbars").joinpath("data")
    train_kwargs = {"batch_size": train_batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    if device.type == "cuda" and torch.cuda.is_available():
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    dataset_train = torchvision.datasets.MNIST(
        str(home_dir), train=True, download=True, transform=transform
    )

    dataset_test = torchvision.datasets.MNIST(str(home_dir), train=False, transform=transform)

    train_loader = DataLoader(dataset_train, **train_kwargs)
    test_loader = DataLoader(dataset_test, **test_kwargs)

    return train_loader, test_loader
