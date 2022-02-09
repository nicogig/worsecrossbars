"""layers:
A backend module dedicated to the creation of custom synaptic layers for TensorFlow.

Comment: Please do not touch this in any way, I'm still working on it!
- nicogig
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import constraints
from tensorflow.keras import layers
from tensorflow.keras import models

import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, device, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class DenseMemristorLayer(layers.Layer):
    def __init__(self, neurons_in: int, neurons_out: int, **kwargs) -> None:
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        super().__init__(**kwargs)
