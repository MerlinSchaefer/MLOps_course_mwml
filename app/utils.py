import json
import random
from typing import Any, Dict

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Set seed for reproducibility.
    Args:
        seed (int, optional): number to be used as the seed. Defaults to 42.
    """
    np.random.seed(seed)
    random.seed(seed)


def replace_dash(x: str) -> str:
    """Replace dashes from tags and aliases
    Args:
        x (str) string to replace dashes from.
    Returns:
        str: string with dashes replaced.
    """
    return x.replace("-", " ")


def save_dict(d: Dict, filepath: str, cls: Any = None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.
    Args:
        d (Dict): data to save.
        filepath (str): location of where to save the data.
        cls (Any,optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.
    Args:
        filepath (str): location of file.
    Returns:
        Dict: loaded JSON data.
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d
