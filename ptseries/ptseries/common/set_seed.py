from numba import njit
import numpy as np
import os
import torch
import random


def set_seed(value=0):
    """Fixes a random seed

    Args:
        value (int): the random seed to use
    """
    os.environ["PYTHONHASHSEED"] = str(value)

    _set_seed_numba(value)  # for Numba, used in the PT Series simulation
    np.random.seed(value)  # for numpy
    random.seed(value)  # for Python
    torch.manual_seed(value)  # for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@njit
def _set_seed_numba(value):
    np.random.seed(value)
