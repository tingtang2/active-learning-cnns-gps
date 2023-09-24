# Neural process families
from torch import nn

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from models.base_modules import MLP

import torch.nn.functional as F

from gpytorch.kernels import RBFKernel, ScaleKernel


class PowerFunction(nn.Module):

    def __init__(self, K=1):
        super().__init__()
        self.K = K

    def forward(self, x):
        return torch.cat(list(map(x.pow, range(self.K + 1))), -1)


# CNP code from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/models/convcnp1d.py


class ConvCNP1d(nn.Module):

    def __init__(self, density=16):
        super().__init__()

        self.density = density

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(nn.Conv1d(6,
                                           16,
                                           5,
                                           1,
                                           2),
                                 nn.ReLU(),
                                 nn.Conv1d(16,
                                           32,
                                           5,
                                           1,
                                           2),
                                 nn.ReLU(),
                                 nn.Conv1d(32,
                                           16,
                                           5,
                                           1,
                                           2),
                                 nn.ReLU(),
                                 nn.Conv1d(16,
                                           2,
                                           5,
                                           1,
                                           2))

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())

    def forward(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor):
        tmp = torch.cat([xc, xt], 1)
        lower, upper = tmp.min().item(), tmp.max().item()
        num_t = int((self.density * (upper - lower)))
        t = torch.linspace(start=lower, end=upper, steps=num_t).reshape(1, -1, 1).repeat(xc.size(0), 1, 4).to(xc.device)

        h = self.psi(t, xc).matmul(self.phi(yc))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)
        h = torch.cat([h0, h1], -1)

        rep = torch.cat([t, h], -1).transpose(-1, -2)
        f = self.cnn(rep).transpose(-1, -2)
        f_mu, f_sigma = f.split(1, -1)

        mu = self.psi_rho(xt, t).matmul(f_mu).squeeze()

        sigma = self.psi_rho(xt, t).matmul(self.pos(f_sigma)).squeeze()
        return MultivariateNormal(mu, scale_tril=sigma.diag_embed())


def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):

        def forward(self, input):
            return F.conv1d(
                input,
                self.weight.abs(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv


# custom ConvCNP for splicing data
class SplicingConvCNP1d(nn.Module):

    def __init__(self, inducer_net: nn.Module, y_dim: int = 5):
        super(SplicingConvCNP1d, self).__init__()

        # first conv layer must produce positive vals to be intepreted as density
        self.initial_conv = make_abs_conv(
            nn.Conv1d(in_channels=y_dim,
                      out_channels=y_dim,
                      groups=y_dim,
                      kernel_size=11,
                      padding=11 // 2,
                      bias=False))

        self.encoder = nn.Sequential(self.initial_conv, inducer_net)
        self.decoder = MLP()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)