from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, Tensor
import torch

import numpy as np
from numpy import ndarray
from pathlib import Path

from src.configs import PATH_TO_DIRECTORY
from src.data.old_dataset import get_dataset

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

def get_splits():
    X, y = get_dataset()
    
    train_indicies = np.load(Path(PATH_TO_DIRECTORY, 'old_data', 'data_indicies', 'trainindices1.npy'))
    test_indicies = np.load(Path(PATH_TO_DIRECTORY, 'old_data', 'data_indicies', 'testindices1.npy'))
    
    X_train, y_train = X[train_indicies], y[train_indicies]
    X_test, y_test = X[test_indicies], y[test_indicies]
        
    print(f'Train X dimensions: {X_train.shape} Test X dimensions: {X_test.shape}')

    return X_train, y_train, X_test, y_test

def create_dataloaders(X_train, y_train, X_test, y_test, device: torch.device, batch_size: int = 128, test_batch_size: int = None):
    '''
        Loads dataset and preprocesses, returns torch dataloaders for 
        training and eval.
    '''
    # set up dataset objects
    train_dataset = FivePSplicingDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device))

    # set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_test_dataloader(X_test, y_test, device, test_batch_size)

    return train_loader, test_loader, X_train.shape[1]


def create_test_dataloader(X_test, y_test, device: torch.device, batch_size = None):
    '''
        Loads dataset and preprocesses, returns torch dataloaders for 
        training and eval.
    '''
    # set up dataset objects
    test_dataset = FivePSplicingDataset(torch.from_numpy(X_test).to(device), torch.from_numpy(y_test).to(device))
    
    if not batch_size:
        test_batch_size = len(test_dataset)
    else:
        test_batch_size = batch_size

    # set up data loaders
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader