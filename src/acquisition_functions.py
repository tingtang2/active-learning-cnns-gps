import numpy as np
import torch

rng = np.random.default_rng(11202022)

from typing import List

from data.data_loader import create_test_dataloader


def random(pool_points: List,
           X_train,
           y_train,
           model,
           device,
           criterion,
           acquisition_batch_size=1,
           **kwargs) -> np.ndarray:
    return rng.choice(np.array(pool_points), acquisition_batch_size, replace=False)


def max_variance(pool_points: np.ndarray,
                 X_train,
                 y_train,
                 model,
                 criterion,
                 device,
                 pool_sample_size,
                 acquisition_batch_size=1,
                 **kwargs) -> np.ndarray:

    pool_sample = rng.choice(pool_points, pool_sample_size, replace=False)

    X_pool_data = X_train[pool_sample]
    y_pool_data = y_train[pool_sample]

    pool_dataloader = create_test_dataloader(X_pool_data, y_pool_data, device)
    pool_mse, pool_var = eval(model=model, loader=pool_dataloader, criterion=criterion, device=device, **kwargs)

    return pool_sample[torch.argsort(pool_var, descending=True)[:acquisition_batch_size].cpu().numpy()]


def expected_improvement():
    pass