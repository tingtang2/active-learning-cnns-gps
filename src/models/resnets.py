# inspired and adapted from ChromDragoNN https://github.com/kundajelab/ChromDragoNN/blob/master/model_zoo/stage1/resnet.py

import torch
import torch.nn.functional as F
from torch import nn


class L1Block(nn.Module):

    def __init__(self):
        super(L1Block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L2Block(nn.Module):

    def __init__(self):
        super(L2Block, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(128, 128, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L3Block(nn.Module):

    def __init__(self):
        super(L3Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.conv2 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(200, 200, (3, 1), stride=(1, 1), padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(200)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)

        self.layer = nn.Sequential(self.conv1,
                                   self.bn1,
                                   nn.ReLU(inplace=True),
                                   self.conv2,
                                   self.bn2,
                                   nn.ReLU(inplace=True),
                                   self.conv3,
                                   self.bn3)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class L4Block(nn.Module):

    def __init__(self):
        super(L4Block, self).__init__()
        self.conv1 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn1 = nn.BatchNorm2d(200)
        self.conv2 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(3, 0))
        self.bn2 = nn.BatchNorm2d(200)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, seq_len: int = 101, dropout_prob: float = 0.15, output_dim: int = 1, blocks=[1, 1, 1, 1]):
        super(ResNet, self).__init__()

        self.dropout = dropout_prob
        self.seq_len = seq_len

        self.conv1 = nn.Conv2d(4, 48, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 64, (3, 1), stride=(1, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(64)
        self.prelayer = nn.Sequential(self.conv1,
                                      self.bn1,
                                      nn.ReLU(inplace=True),
                                      self.conv2,
                                      self.bn2,
                                      nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(*[L1Block() for x in range(blocks[0])])
        self.layer2 = nn.Sequential(*[L2Block() for x in range(blocks[1])])
        self.layer3 = nn.Sequential(*[L3Block() for x in range(blocks[2])])
        self.layer4 = nn.Sequential(*[L4Block() for x in range(blocks[3])])

        self.c1to2 = nn.Conv2d(64, 128, (3, 1), stride=(1, 1), padding=(1, 0))
        self.b1to2 = nn.BatchNorm2d(128)
        self.l1tol2 = nn.Sequential(self.c1to2, self.b1to2, nn.ReLU(inplace=True))

        self.c2to3 = nn.Conv2d(128, 200, (1, 1), padding=(3, 0))
        self.b2to3 = nn.BatchNorm2d(200)
        self.l2tol3 = nn.Sequential(self.c2to3, self.b2to3, nn.ReLU(inplace=True))

        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(400, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, output_dim)
        self.flayer = self.final_layer()

    def final_layer(self):
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(200)
        return nn.Sequential(self.conv3, self.bn3, nn.ReLU(inplace=True))

    def forward(self, s):
        s = s.permute(0, 2, 1).contiguous()    # batch_size x 4 x 1000
        s = s.view(-1, 4, self.seq_len, 1)    # batch_size x 4 x 1000 x 1 [4 channels]

        out = self.prelayer(s)
        out = self.layer1(out)
        out = self.layer2(self.l1tol2(out))
        out = self.maxpool1(out)
        out = self.layer3(self.l2tol3(out))
        out = self.maxpool2(out)
        out = self.layer4(out)
        out = self.flayer(out)
        out = self.maxpool3(out)
        out = out.view(-1, 400)
        conv_out = out
        out = F.dropout(F.relu(self.bn4(self.fc1(out))), p=self.dropout, training=self.training)    # batch_size x 1000
        out = F.dropout(F.relu(self.bn5(self.fc2(out))), p=self.dropout, training=self.training)    # batch_size x 1000
        out = self.fc3(out)
        return out
