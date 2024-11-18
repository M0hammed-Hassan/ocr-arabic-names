import os
import torch
import random
import numpy as np

DEFAULT_RANDOM_SEED = 42


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    """
    Seeds the random number generators for reproducibility.

    Parameters:
        seed (int): The seed value to use for random number generation. Defaults to DEFAULT_RANDOM_SEED.

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def seed_torch(seed=DEFAULT_RANDOM_SEED):
    """
    Set the random seed for torch operations.

    Parameters:
        seed (int): The random seed to be set. Defaults to DEFAULT_RANDOM_SEED.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    """
    Seeds all random number generators for reproducibility.

    Parameters:
        seed (int): The seed value to use for random number generation. Defaults to DEFAULT_RANDOM_SEED.

    Returns:
        None
    """
    seed_basic(seed)
    seed_torch(seed)
