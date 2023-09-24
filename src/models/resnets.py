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


# code from https://github.com/cambridge-mlg/convcnp/blob/master/convcnp/utils.py


def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.

    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.

    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:    # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:    # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)), 'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:    # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:    # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)), 'reflect')

    return torch.cat([t1, t2], dim=1)


def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.

    Args:
        layer (:class:`nn.Module`): Single dense or convolutional layer from
            :mod:`torch.nn`.

    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)


class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.

    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = 16
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5,
                                     stride=2,
                                     padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5,
                                      stride=2,
                                      padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5,
                                      stride=2,
                                      padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5,
                                      stride=2,
                                      padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)
