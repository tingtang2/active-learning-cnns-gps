import torch
from torch import nn


class DeepFactorizedModel(nn.Module):

    def __init__(self) -> None:
        super(DeepFactorizedModel, self).__init__()

    def forward(x: torch.Tensor):
        return x