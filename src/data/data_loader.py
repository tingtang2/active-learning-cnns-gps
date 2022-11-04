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

def build_dataloaders(batch_size: int = 128, use_old_split: bool = True):
    '''
        Loads dataset and preprocesses, returns torch dataloaders for 
        training and eval.
    '''
    X, y = get_dataset()
    
    if use_old_split:
        train_indicies = np.load(Path(PATH_TO_DIRECTORY, 'old_data', 'data_indicies', 'trainindices1.npy'))
        test_indicies = np.load(Path(PATH_TO_DIRECTORY, 'old_data', 'data_indicies','testindices1.npy'))
    else:
        train_indicies = None
        test_indicies = None
    
    X_train, y_train = X[train_indicies], y[train_indicies]
    X_test, y_test = X[test_indicies], y[test_indicies]
        
    print(f'Train X dimensions: {X_train.shape} Test X dimensions: {X_test.shape}')
    
    # set up dataset objects
    train_dataset = FivePSplicingDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = FivePSplicingDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, X.shape[1]
