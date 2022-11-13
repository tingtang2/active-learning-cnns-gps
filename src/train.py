import argparse
import sys

from src.models.base_cnn import BaseCNN
from src.data.data_loader import get_splits, create_dataloaders
from src import acquisition_functions

from torch.optim import SGD, RMSprop
from torch.nn import MSELoss
import torch
from torch.utils.data import DataLoader

from tqdm import trange
import plotly.express as px
import numpy as np

from typing import List
from pathlib import Path

import pandas as pd
import json


def active_train(model_type, 
          optimizer_type, 
          criterion, 
          X_train,
          y_train,
          X_test,
          y_test,
          device,
          acquisition_fn,
          begin_train_set_size,
          num_acquisitions,
          l2_penalty,
          **kwargs) -> List[float]:
    
    train_pool = [i for i in range(begin_train_set_size)]
    acquisition_pool = [i for i in range(begin_train_set_size, X_train.shape[0])]
    mse = []
    
    # experiment setup stuff
    for round in trange(num_acquisitions):
        X_train_data = X_train[train_pool]
        y_train_data = y_train[train_pool]
        
        model = model_type().double().to(device)
        optimizer = optimizer_type(model.parameters(), lr=0.001, weight_decay=l2_penalty/len(train_pool))
        
        train_dataloader, test_dataloader, num_feats = create_dataloaders(X_train=X_train_data, y_train=y_train_data, X_test=X_test, y_test=y_test, device=device)
        new_mse, new_var = active_iteration(model=model, 
                                            train_loader=train_dataloader, 
                                            test_loader=test_dataloader, 
                                            optimizer=optimizer, 
                                            criterion=criterion, 
                                            device=device, 
                                            **kwargs)
        
        mse.append(new_mse.item())

        new_points = acquisition_fn(pool_points=acquisition_pool, 
                                    X_train=X_train, 
                                    y_train=y_train, 
                                    model=model)
        for point in new_points:
            train_pool.append(point)
            acquisition_pool.remove(point)
    return mse

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def eval(model,
         loader: DataLoader,
         criterion,
         mc_dropout_iterations: int,
         device: torch.device):
    model.eval()
    enable_dropout(model)
    
    predictions = torch.empty((mc_dropout_iterations, loader.dataset.sequences.size(0))).to(device)
    full_labels: torch.Tensor = loader.dataset.proportions

    with torch.no_grad():
        for i in range(mc_dropout_iterations):
            for j, batch in enumerate(loader):
                examples, labels = batch     
            
                predictions[i] = model(examples).reshape(-1)
    pred_var, preds = torch.var_mean(predictions, dim=0)

    return criterion(preds, full_labels.double()), pred_var


# pass in updated train and test loaders at each iteration
def active_iteration(model: torch.nn.Module,
                     train_loader: DataLoader,
                     test_loader: DataLoader,
                     optimizer,
                     criterion,
                     epochs: int,
                     mc_dropout_iterations: int,
                     device: torch.device,
                     **kwargs):
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()    
            examples, labels = batch     
        
            predictions = model(examples).reshape(-1)

            loss = criterion(predictions, labels.double())

            loss.double().backward()
            optimizer.step()

            running_loss += loss.item()

        #print(f'epoch {epoch} loss: {running_loss}')

    return eval(model, test_loader, criterion, mc_dropout_iterations, device=device)

def plot(name: str, metrics: List[float], save_dir: str, **kwargs) -> None:
    df = pd.DataFrame({name: [item for item in metrics], 'iteration': [i for i in range(1, len(metrics) + 1)]})
    fig = px.line(df, x='iteration', y=name, title=f'{name} as a Function of Iteration')
    fig.write_html(Path(Path.home(), save_dir, 'fig.html'))
    fig.write_image(Path(Path.home(), save_dir, 'fig.png'))


def main() -> int:

    configs = {
        'epochs': 100,
        'batch_size': 128,
        'num_acquisitions': 100,
        'acquisition_batch_size': 1,
        'sample_size': 5000,
        'mc_dropout_iterations': 10,
        'size_train': 75,
        'tau_inv_proportion': 0.15,
        'begin_train_set_size': 75,
        'l2_penalty': 0.025,
        'save_dir': 'saved_metrics/',
        'acquisition_fn_type': 'random'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, y_train, X_test, y_test = get_splits()
    
    # experiment setup stuff
    criterion = MSELoss()
    if configs['acquisition_fn_type'] == 'random':
        acquisition_fn = acquisition_functions.random

    mse = active_train(model_type=BaseCNN, 
                       optimizer_type=RMSprop, 
                       criterion=criterion, 
                       X_train=X_train, 
                       y_train=y_train, 
                       X_test=X_test, 
                       y_test=y_test, 
                       device=device, 
                       acquisition_fn=acquisition_fn,  
                       **configs) 

    plot('Mean Square Error', mse, **configs)
    with open(Path(Path.home(), configs['save_dir'], f'{configs["acquisition_fn_type"]}.json'), 'w') as f:
        json.dump(mse, f)


    return 0

if __name__ == '__main__':
    sys.exit(main())