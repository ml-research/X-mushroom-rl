import torch
import torch.nn as nn
from x_mushroom_rl.utils.features import uniform_grid
from x_mushroom_rl.utils.torch import to_float_tensor, to_int_tensor
import matplotlib.pyplot as plt
import numpy as np


class DeepExtractor(nn.Module):
    """
    Pytorch DeepExtractor module that extracts features.

    """
    def __init__(self, input_shape=None, ouput_shape=None):
        """
        Constructor.

        Args:

        """
        super().__init__()
        pass


    def forward(self, x):
        return 0

    @staticmethod
    def generate(n_centers, low, high,  dimensions=None, use_cuda=False):
        import ipdb; ipdb.set_trace()

    @property
    def size(self):
        return None
