# Neural process families
from typing import Union
from torch import nn

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Independent, Normal
from torch.nn.common_types import _size_1_t
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


class AbsConv1d(nn.Conv1d):

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


def MultivariateNormalDiag(loc, scale_diag):
    """Multi variate Gaussian with a diagonal covariance function (on the last dimension)."""
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


# custom ConvCNP for splicing data
class SplicingConvCNP1d(nn.Module):

    def __init__(self,
                 inducer_net: nn.Module,
                 device: torch.device,
                 dropout: float = 0.15,
                 x_dim: int = 5,
                 r_dim: int = 128,
                 seq_len: int = 101):
        super(SplicingConvCNP1d, self).__init__()

        # first conv layer must produce positive vals to be intepreted as density
        self.initial_conv = AbsConv1d(in_channels=x_dim,
                                      out_channels=x_dim,
                                      groups=x_dim,
                                      kernel_size=11,
                                      padding=11 // 2,
                                      bias=False)

        self.resizer = nn.Linear(2 * x_dim, r_dim)
        self.encoder = inducer_net

        # TODO: think of a better way to set input dimensionality
        input_dim = -1
        if seq_len == 101:
            input_dim = 82324
        elif seq_len == 109:
            input_dim == 410004

        self.decoder = nn.Sequential(MLP(n_in=input_dim,
                                         n_out=32,
                                         dropout=dropout),
                                     nn.Linear(in_features=32,
                                               out_features=2))

        self.device = device

    def forward(self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor):
        # batch, seq_len, 4 + 1
        x_c = x_c.transpose(-1, -2)
        y_c_channel = y_c.repeat(x_c.size(2), 1, 1)

        # append y and 1, but I don't think we need density channel because we're not making a prediction at each location
        context_set = torch.cat((x_c, y_c_channel.view(x_c.size(0), 1, -1)), dim=1)
        density = torch.ones(context_set.size()).to(self.device)

        func_rep = self.initial_conv(context_set)
        density = self.initial_conv(density)

        func_rep = func_rep / torch.clamp(density, min=1e-5)    # normalize convolution

        func_rep = torch.cat((func_rep, density), dim=1).transpose(-1, -2)

        func_rep = self.resizer(func_rep)

        func_rep = self.encoder(func_rep.transpose(-1, -2))

        if x_t.size(0) != 128:
            print(x_t.size(), func_rep.size())

        final_rep = torch.cat((func_rep.view(x_t.size(0), -1), x_t.view(x_t.size(0), -1)), dim=-1)

        mu, sigma = self.decoder(final_rep).split(1, dim=-1)
        sigma = 0.01 + 0.99 * F.softplus(sigma)

        return MultivariateNormalDiag(loc=mu, scale_diag=sigma)
