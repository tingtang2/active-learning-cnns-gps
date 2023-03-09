import torch
from torch import nn
from torch.nn import functional as F


class DeepFactorizedModel(nn.Module):

    def __init__(self, seq_len: int = 101, dropout_prob: float = 0.15, output_dim: int = 1) -> None:
        super(DeepFactorizedModel, self).__init__()

        self.dropout = dropout_prob
        self.seq_len = seq_len

        self.layer1 = self.layer_one()
        self.layer2 = self.layer_two()
        self.layer3 = self.layer_three()
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(400, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, output_dim)

    def layer_one(self):
        self.conv1a = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1b = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1c = nn.Conv2d(64, 100, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv1d = nn.Conv2d(100, 150, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv1e = nn.Conv2d(150, 300, (7, 1), stride=(1, 1), padding=(3, 0))

        self.bn1a = nn.BatchNorm2d(48)
        self.bn1b = nn.BatchNorm2d(64)
        self.bn1c = nn.BatchNorm2d(100)
        self.bn1d = nn.BatchNorm2d(150)
        self.bn1e = nn.BatchNorm2d(300)

        tmp = nn.Sequential(self.conv1a,
                            self.bn1a,
                            nn.ReLU(inplace=True),
                            self.conv1b,
                            self.bn1b,
                            nn.ReLU(inplace=True),
                            self.conv1c,
                            self.bn1c,
                            nn.ReLU(inplace=True),
                            self.conv1d,
                            self.bn1d,
                            nn.ReLU(inplace=True),
                            self.conv1e,
                            self.bn1e,
                            nn.ReLU(inplace=True))

        return tmp

    def layer_two(self):
        self.conv2a = nn.Conv2d(300, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2b = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv2c = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))

        self.bn2a = nn.BatchNorm2d(200)
        self.bn2b = nn.BatchNorm2d(200)
        self.bn2c = nn.BatchNorm2d(200)

        tmp = nn.Sequential(self.conv2a,
                            self.bn2a,
                            nn.ReLU(inplace=True),
                            self.conv2b,
                            self.bn2b,
                            nn.ReLU(inplace=True),
                            self.conv2c,
                            self.bn2c,
                            nn.ReLU(inplace=True))

        return tmp

    def layer_three(self):
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(200)
        return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))

    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()    # batch_size x 4 x 1000
        s = s.view(-1, 4, self.seq_len, 1)    # batch_size x 4 x 1000 x 1 [4 channels]
        s = self.maxpool1(self.layer1(s))    # batch_size x 300 x 333 x 1
        s = self.maxpool2(self.layer2(s))    # batch_size x 200 x 83 x 1
        s = self.maxpool3(self.layer3(s))    # batch_size x 200 x 21 x 1
        s = s.view(-1, 400)
        conv_out = s
        s = F.dropout(F.relu(self.bn4(self.fc1(s))), p=self.dropout, training=self.training)    # batch_size x 1000
        s = F.dropout(F.relu(self.bn5(self.fc2(s))), p=self.dropout, training=self.training)    # batch_size x 1000
        s = self.fc3(s)
        return s