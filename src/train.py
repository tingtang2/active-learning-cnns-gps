import argparse
from datetime import date

import logging
import sys

from src.models.base_cnn import BaseCNN
from src.models.dkl import ApproximateDKLRegression
from src.data.data_loader import get_splits, create_dataloaders
from src import acquisition_functions
from src.eval import eval

import torch
import gpytorch
from torch.optim import SGD, RMSprop
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(11202022)

from tqdm import trange, tqdm
import plotly.express as px

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
          acquisition_fn_type,
          begin_train_set_size,
          num_acquisitions,
          l2_penalty,
          **kwargs) -> List[float]:
    
    train_pool = [i for i in range(begin_train_set_size)]
    acquisition_pool = [i for i in range(begin_train_set_size, X_train.shape[0])]
    mse = []

    if 'dkl' in acquisition_fn_type:
        for round in trange(num_acquisitions):
            X_train_data = X_train[train_pool]
            y_train_data = y_train[train_pool]
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = model_type().double().to(device)
            optimizer = optimizer_type(model.parameters(), lr=0.001, weight_decay=l2_penalty/len(train_pool))
            optimizer = SGD([
                {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
                {'params': model.gp_layer.hyperparameters(), 'lr': 0.001 * 0.01},
                {'params': model.gp_layer.variational_parameters()},
                {'params': likelihood.parameters()},
            ], lr=0.001, momentum=0.9, nesterov=True, weight_decay=0)
            
            train_dataloader, test_dataloader, num_feats = create_dataloaders(X_train=X_train_data, y_train=y_train_data, X_test=X_test, y_test=y_test, device=device)
            # scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
            mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_dataloader.dataset))

            
            new_mse, new_var = active_iteration_dkl(model=model, 
                                                train_loader=train_dataloader, 
                                                test_loader=test_dataloader, 
                                                optimizer=optimizer, 
                                                criterion=criterion,
                                                likelihood=likelihood,
                                                mll=mll, 
                                                device=device, 
                                                **kwargs)
            
            mse.append(new_mse.item())

            new_points = acquisition_fn(pool_points=acquisition_pool, 
                                        X_train=X_train, 
                                        y_train=y_train, 
                                        model=model, 
                                        criterion=criterion,
                                        device=device,
                                        **kwargs)

            for point in new_points:
                train_pool.append(point)
                acquisition_pool.remove(point)

    else:
        for round in trange(num_acquisitions):
            X_train_data = X_train[train_pool].astype(np.float32)
            y_train_data = y_train[train_pool].astype(np.float32)
            
            model = model_type().to(device)
            # extra insurance
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

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
            logging.info(f'AL iteration: {round + 1}, MSE: {mse[-1]}, len train set: {len(train_pool)} len acquisition pool {len(acquisition_pool)}')

            new_points = acquisition_fn(pool_points=acquisition_pool, 
                                        X_train=X_train, 
                                        y_train=y_train, 
                                        model=model, 
                                        criterion=criterion,
                                        device=device,
                                        **kwargs)

            for point in new_points:
                logging.info(f'index added {point}')
                train_pool.append(point)
                acquisition_pool.remove(point)
    return mse

def active_iteration_dkl(model, likelihood, epochs, optimizer, train_loader, test_loader, mll, mc_dropout_iterations, **kwargs):
    for epoch in range(1, epochs + 1):
        with gpytorch.settings.use_toeplitz(False):
            train_dkl(epoch, model, likelihood, optimizer, train_loader, mll)
            # scheduler.step()
    return eval_dkl(model, likelihood, test_loader, mc_dropout_iterations)

def train_dkl(epoch, model, likelihood, optimizer, train_loader, mll):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for batch in minibatch_iter:
            examples, labels = batch     
            optimizer.zero_grad()
            output = model(examples)
            loss = -mll(output, )
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())

def eval_dkl(model, likelihood, test_loader, mc_dropout_iterations):
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(mc_dropout_iterations):
        for data, target in test_loader:
            preds = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
    print(preds)
    print(dir(preds))
    return preds.mean(), preds.variance()


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

            loss = criterion(predictions, labels)

            loss.double().backward()
            optimizer.step()

            running_loss += loss.item()

    return eval(model, test_loader, criterion, mc_dropout_iterations, device=device)

def plot(name: str, metrics: List[float], save_dir: str, acquisition_fn_type: str, **kwargs) -> None:
    df = pd.DataFrame({name: [item for item in metrics], 'iteration': [i for i in range(1, len(metrics) + 1)]})
    fig = px.line(df, x='iteration', y=name, title=f'{acquisition_fn_type}: {name} as a Function of Iteration')
    fig.write_html(Path(Path.home(), save_dir, f'{acquisition_fn_type}.html'))
    fig.write_image(Path(Path.home(), save_dir, f'{acquisition_fn_type}.png'))


def main() -> int:
    # filename for logging and saved models
    configs = {
        'epochs': 300,
        'batch_size': 128,
        'num_acquisitions': 600,
        'acquisition_batch_size': 1,
        'pool_sample_size': 5000,
        'mc_dropout_iterations': 50,
        'size_train': 75,
        'tau_inv_proportion': 0.15,
        'begin_train_set_size': 75,
        'l2_penalty': 0.025,
        'save_dir': 'saved_metrics/',
        'acquisition_fn_type': 'random',
        'num_repeats': 3
    }
    filename = f'al-{configs["acquisition_fn_type"]}-{date.today()}_test'

    logging.basicConfig(level=logging.DEBUG, filename= './' + filename+'.log', filemode='a', format='%(message)s')
    logging.info(configs)
    
    for iter in range(configs['num_repeats']):
        # get device and data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train, y_train, X_test, y_test = get_splits()
        
        # experiment setup stuff
        model = BaseCNN
        criterion = MSELoss()
        
        if configs['acquisition_fn_type'] == 'random':
            acquisition_fn = acquisition_functions.random
        elif configs['acquisition_fn_type'] == 'max_variance':
            acquisition_fn = acquisition_functions.max_variance
        elif configs['acquisition_fn_type'] == 'max_variance_dkl':
            acquisition_fn = acquisition_functions.max_variance
            model = ApproximateDKLRegression
        

        # run active learning experiment
        mse = active_train(model_type=model, 
                        optimizer_type=RMSprop, 
                        criterion=criterion, 
                        X_train=X_train, 
                        y_train=y_train, 
                        X_test=X_test, 
                        y_test=y_test, 
                        device=device, 
                        acquisition_fn=acquisition_fn,  
                        **configs) 

        # plot and save
        #plot(name='Mean Square Error', metrics=mse, **configs)
        with open(Path(Path.home(), configs['save_dir'], f'{configs["acquisition_fn_type"]}_iteration_{iter}.json'), 'w') as f:
            json.dump(mse, f)

    return 0

if __name__ == '__main__':
    sys.exit(main())