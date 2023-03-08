import torch
from torch import nn


class MPRADragonNN(nn.Module):

    def __init__(self) -> None:
        super(MPRADragonNN, self).__init__()

    def forward(x: torch.Tensor):
        return x