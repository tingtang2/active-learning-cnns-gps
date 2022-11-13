import argparse
import sys

from torch.utils.data import DataLoader
import torch

from typing import Tuple

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def eval(model,
         loader: DataLoader,
         criterion,
         mc_dropout_iterations: int,
         device: torch.device,
         **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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

def main() -> int:
    return 0

if __name__ == '__main__':
    sys.exit(main())