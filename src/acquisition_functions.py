import torch
import numpy as np
from typing import List

def random(pool_points: np.ndarray, X_train, y_train, model, batch_size=1) -> np.ndarray:
    return np.random.choice(pool_points, batch_size)

def max_variance(pool_points: np.ndarray, X_train, y_train, model, batch_size=1) -> np.ndarray:
    return 

def expected_improvement():
    pass