import argparse
import sys
from src.models.base_cnn import BaseCNN
from src.data.data_loader import get_splits, create_dataloaders
from torch.optim import SGD, RMSprop
from torch.nn import MSELoss
import torch
from torch.utils.data import DataLoader
from tqdm import trange

def active_train(model, 
          optimizer, 
          criterion, 
          data_loader,
          acquisition_function):
    
    pass

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def eval(model,
         loader,
         criterion,
         mc_dropout_iterations: int):
    model.eval()
    enable_dropout(model)
    
    predictions = torch.empty((mc_dropout_iterations, 26513))
    full_labels: torch.Tensor = torch.empty([])

    with torch.no_grad():
        for i in trange(mc_dropout_iterations):
            for j, batch in enumerate(loader):
                examples, labels = batch     
            
                predictions[i] = model(examples).reshape(-1)
                if i == 0:
                    full_labels = labels
    preds = torch.mean(predictions, dim=0).reshape(-1)

    return criterion(preds, full_labels.double())


# pass in updated train and test loaders at each iteration
def active_iteration(model: torch.nn.Module,
                     train_loader: DataLoader,
                     test_loader: DataLoader,
                     optimizer,
                     criterion,
                     epochs: int,
                     mc_dropout_iterations: int,
                     **kwargs):
    
    model.train()
    for epoch in trange(epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()    
            examples, labels = batch     
        
            predictions = model(examples).reshape(-1)

            loss = criterion(predictions, labels.double())

            loss.double().backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'epoch loss: {running_loss}')

    return eval(model, test_loader, criterion, mc_dropout_iterations)


def main() -> int:

    configs = {
        'epochs': 100,
        'batch_size': 128,
        'num_acquisitions': 1000,
        'acquisition_batch_size': 1,
        'sample_size': 5000,
        'num_epochs': 300,
        'acq_fn_dropout_iterations': 50,
        'mc_dropout_iterations': 100,
        'size_train': 75,
        'tau_inv_proportion': 0.15,
        'begin_train_set_size': 75,
        'l2_penalty': 0.025
    }

    X_train, y_train, X_test, y_test = get_splits()
    model = BaseCNN().double()
    print('initialized model')

    # experiment setup stuff
    criterion = MSELoss()
    optimizer = RMSprop(model.parameters(), lr=0.001, weight_decay=configs['l2_penalty']/configs['begin_train_set_size'])

    mse = [] # list to hold metrics

    # first iteration of active learning experiment
    X_train_data = X_train[:configs['begin_train_set_size']]
    y_train_data = y_train[:configs['begin_train_set_size']]

    train_dataloader, test_dataloader, num_feats = create_dataloaders(X_train=X_train_data, y_train=y_train_data, X_test=X_test, y_test=y_test)

    print(active_iteration(model=model, train_loader=train_dataloader, test_loader=test_dataloader, optimizer=optimizer, criterion=criterion, **configs))


    # y_hat = model(torch.from_numpy(X[:10]))
    # print(criterion(y_hat.squeeze(), torch.from_numpy(y[:10])))


    return 0

if __name__ == '__main__':
    sys.exit(main())