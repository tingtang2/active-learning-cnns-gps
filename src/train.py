import argparse
import sys
from src.models.base_cnn import BaseCNN
from src.data.data_loader import get_splits, build_dataloaders
from torch.optim import SGD
from torch.nn import MSELoss
import torch
from torch.utils.data import DataLoader

def train(model, 
          optimizer, 
          criterion, 
          data_loader):
    raise NotImplementedError

def active_iteration(model, 
                     train_loader: DataLoader, 
                     **kwargs):
    
    
    model.fit(X_train_rand, y_train_rand, epochs=num_epochs, batch_size=batchsize, verbose=0)
    f_rand = K.function([model_rand.layers[0].input, K.learning_phase()], 
                [model_rand.layers[-1].output])
    predictions_with_uncertainty = predict_with_uncertainty(f_rand, X_test, n_iter=mse_dropout_iterations)
    y_predicted = predictions_with_uncertainty[0]
    mse = np.mean(np.square(y_test-y_predicted))


def main() -> int:

    configs = {
        'epochs': 100,
        'batch_size': 128,
        'num_acquisitions': 1000,
        'acquisition_batch_size': 1,
        'sample_size': 5000,
        'num_epochs': 300,
        'acq_fn_dropout_iterations': 50,
        'mse_dropout_iterations': 200,
        'size_train': 75,
        'tau_inv_proportion': 0.15
    }

    train_loader, test_loader, shape = build_dataloaders()
    X_train, y_train, X_test, y_test = get_splits()
    model = BaseCNN().double()
    print('initialized model')

    criterion = MSELoss()

    # y_hat = model(torch.from_numpy(X[:10]))
    # print(criterion(y_hat.squeeze(), torch.from_numpy(y[:10])))


    return 0

if __name__ == '__main__':
    sys.exit(main())