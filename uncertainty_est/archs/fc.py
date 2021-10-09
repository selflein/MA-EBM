import torch
from torch import nn
import torch.nn.functional as F

from uncertainty_est.archs.invertible_residual_nets.bound_spectral_norm import (
    spectral_norm_fc,
)


class NegativeLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, input):
        return F.linear(input, -torch.exp(self.weight), self.bias)


class RBF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return torch.exp(-torch.pow(inp, 2))


class MReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return torch.min(F.relu(1 - inp), F.relu(1 + inp))


def make_mlp(
    dim_list,
    activation="relu",
    batch_norm=False,
    dropout=0,
    bias=True,
    slope=1e-2,
    spectral_norm_kwargs=None,
    neg_linear=False,
):
    layers = []
    if len(dim_list) > 2:
        for dim_in, dim_out in zip(dim_list[:-2], dim_list[1:-1]):
            if spectral_norm_kwargs is not None:
                layers.append(
                    spectral_norm_fc(
                        nn.Linear(dim_in, dim_out, bias=bias), **spectral_norm_kwargs
                    )
                )
            else:
                layers.append(nn.Linear(dim_in, dim_out, bias=bias))

            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out, affine=True))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "rbf":
                layers.append(RBF())
            elif activation == "mrelu":
                layers.append(MReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(slope, inplace=True))
            elif activation == "elu":
                layers.append(nn.ELU(inplace=True))
            else:
                raise NotImplementedError(f"Activation '{activation}' not implemented!")

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

    if neg_linear:
        last_linear = NegativeLinear(dim_list[-2], dim_list[-1], bias=bias)
    else:
        last_linear = nn.Linear(dim_list[-2], dim_list[-1], bias=bias)

    if spectral_norm_kwargs is not None:
        layers.append(spectral_norm_fc(last_linear, **spectral_norm_kwargs))
    else:
        layers.append(last_linear)
    model = nn.Sequential(*layers)
    return model


class SynthModel(nn.Module):
    def __init__(
        self,
        inp_dim,
        num_classes,
        hidden_dims=[
            50,
            50,
        ],
        activation="leaky_relu",
        batch_norm=False,
        dropout=0.0,
        neg_linear=False,
        **kwargs,
    ):
        super().__init__()
        self.net = make_mlp(
            [
                inp_dim,
            ]
            + hidden_dims
            + [
                num_classes,
            ],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
            neg_linear=neg_linear,
            **kwargs,
        )

    def forward(self, inp):
        return self.net(inp)


class LipschitzResNet(nn.Module):
    def __init__(
        self,
        inp_dim,
        num_classes,
        hidden_dims=[50, 50],
        block_hidden_dims=[
            50,
            50,
        ],
        activation="leaky_relu",
        batch_norm=False,
        dropout=0.0,
        coeff=0.9,
        n_power_iterations=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.transforms = nn.ModuleList()
        dims = (
            [
                inp_dim,
            ]
            + hidden_dims
            + [
                num_classes,
            ]
        )
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.blocks.append(
                make_mlp(
                    [in_dim] + block_hidden_dims + [in_dim],
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    spectral_norm_kwargs={
                        "coeff": coeff,
                        "n_power_iterations": n_power_iterations,
                    },
                )
            )
            self.transforms.append(
                spectral_norm_fc(
                    nn.Linear(in_dim, out_dim),
                    coeff,
                    n_power_iterations=n_power_iterations,
                )
            )

    def forward(self, x):
        for block, transf in zip(self.blocks, self.transforms):
            res = block(x)
            x = x + res
            x = transf(x)
        return x
