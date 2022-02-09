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


class DenseMemristorLayer(layers.Layer):
    def __init__(self, neurons_in: int, neurons_out: int, **kwargs) -> None:
        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        super().__init__(**kwargs)
