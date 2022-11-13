import torch
import numpy as np
from typing import List

def random(pool_points: np.ndarray, batch_size=1) -> np.ndarray:
    return np.random.choice(pool_points, batch_size)

def max_variance():
    pass

def expected_improvement():
    pass