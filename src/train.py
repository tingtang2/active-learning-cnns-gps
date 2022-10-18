import argparse
import sys
from src.models.base_cnn import BaseCNN
from src.data.old_dataset import get_dataset
from torch.optim import SGD
from torch.nn import MSELoss
import torch

def train(model, 
          optimizer, 
          criterion, 
          data_loader):
    raise NotImplementedError


def main() -> int:
    X, y = get_dataset()
    model = BaseCNN().double()
    print('initialized model')

    criterion = MSELoss()

    y_hat = model(torch.from_numpy(X[:10]))
    print(criterion(y_hat.squeeze(), torch.from_numpy(y[:10])))


    return 0

if __name__ == '__main__':
    sys.exit(main())