import torch
from torch import nn


class BaseCNN(nn.Module):

    def __init__(self,
                 seq_len: int = 101,
                 dropout_prob: float = 0.15,
                 MLP_out_dim: int = 50,
                 output_dim: int = 1) -> None:
        super(BaseCNN, self).__init__()

        # configs
        self.dropout_prob = dropout_prob
        self.seq_len = seq_len

        # layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.seq_len, kernel_size=4)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=self.seq_len, out_channels=self.seq_len // 2, kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.dense = nn.Linear(in_features=450, out_features=MLP_out_dim)
        self.output = nn.Linear(in_features=MLP_out_dim, out_features=output_dim)

        # easy turning on and off of dropout
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x: torch.Tensor, fixed_mask: torch.Tensor = None) -> torch.Tensor:
        if len(x.size()) < 3:
            x = x.reshape(-1, 101, 4)

        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.reshape((x.size(0), -1))
        x = self.dense(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.output(x)

        return x


class OracleCNN(nn.Module):

    def __init__(self,
                 seq_len: int = 101,
                 dropout_prob: float = 0.15,
                 MLP_out_dim: int = 50,
                 output_dim: int = 1) -> None:
        super(OracleCNN, self).__init__()

        # configs
        self.dropout_prob = dropout_prob
        self.seq_len = seq_len

        # layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.seq_len, kernel_size=4)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=self.seq_len, out_channels=self.seq_len // 2, kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.dense = nn.Linear(in_features=450, out_features=MLP_out_dim)
        self.output = nn.Linear(in_features=MLP_out_dim, out_features=output_dim)

        # easy turning on and off of dropout
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x: torch.Tensor, fixed_mask: torch.Tensor = None) -> torch.Tensor:
        if len(x.size()) < 3:
            x = x.reshape(-1, 101, 4)

        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.reshape((x.size(0), -1))
        x = self.dense(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.output(x)

        return x