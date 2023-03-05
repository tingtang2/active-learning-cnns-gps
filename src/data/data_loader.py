from pathlib import Path

import numpy as np
import torch
from numpy import ndarray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from configs import PATH_TO_DIRECTORY
from data.old_dataset import get_dataset

import logging


class FivePSplicingDataset(Dataset):
    '''
        torch dataset class for easy batching of 5p splicing dataset
    '''

    def __init__(self, sequences: Tensor, proportions: Tensor):
        self.sequences = sequences
        self.proportions = proportions

    def __getitem__(self, idx):
        return self.sequences[idx], self.proportions[idx]

    def __len__(self):
        return len(self.proportions)


def get_splits(iter: int = 1):
    X, y = get_dataset()

    train_idx_path = Path(PATH_TO_DIRECTORY, 'old_data', 'data_indicies', f'trainindices{iter}.npy')
    test_idx_path = Path(PATH_TO_DIRECTORY, 'old_data', 'data_indicies', f'testindices{iter}.npy')
    logging.info(f'train idx path: {train_idx_path}, test idx path: {test_idx_path}')

    train_indicies = np.load(train_idx_path)
    test_indicies = np.load(test_idx_path)

    X_train, y_train = X[train_indicies], y[train_indicies]
    X_test, y_test = X[test_indicies], y[test_indicies]

    print(f'Train X dimensions: {X_train.shape} Test X dimensions: {X_test.shape}')

    return X_train, y_train, X_test, y_test


def get_oracle_splits(seed):
    X, y = get_dataset()

    train_indicies = np.load(Path(PATH_TO_DIRECTORY, 'old_data', 'data_indicies', 'trainindices1.npy'))

    X_not_test, y_not_test = X[train_indicies], y[train_indicies]
    X_train, X_val, y_train, y_val = train_test_split(X_not_test, y_not_test, test_size=26513, seed=seed)

    print(f'Train X dimensions: {X_train.shape} Val X dimensions: {X_val.shape}')

    return X_train, y_train, X_val, y_val


def create_dataloaders(X_train,
                       y_train,
                       X_test,
                       y_test,
                       device: torch.device,
                       batch_size: int = 128,
                       test_batch_size: int = None):
    '''
        Loads dataset and preprocesses, returns torch dataloaders for 
        training and eval.
    '''
    # set up dataset objects
    train_dataset = FivePSplicingDataset(
        torch.from_numpy(X_train).float().to(device),
        torch.from_numpy(y_train).float().to(device))

    # set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_test_dataloader(X_test, y_test, device, test_batch_size)

    return train_loader, test_loader, X_train.shape[1]


def create_test_dataloader(X_test, y_test, device: torch.device, batch_size=None):
    '''
        Loads dataset and preprocesses, returns torch dataloaders for 
        training and eval.
    '''
    # set up dataset objects
    test_dataset = FivePSplicingDataset(
        torch.from_numpy(X_test).to(device,
                                    dtype=torch.float),
        torch.from_numpy(y_test).to(device,
                                    dtype=torch.float))

    if not batch_size:
        test_batch_size = len(test_dataset)
    else:
        test_batch_size = batch_size

    # set up data loaders
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader