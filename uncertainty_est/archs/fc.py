from torch import nn

from uncertainty_est.archs.invertible_residual_nets.bound_spectral_norm import (
    spectral_norm_fc,
)


def make_mlp(
    dim_list,
    activation="relu",
    batch_norm=False,
    dropout=0,
    bias=True,
    slope=1e-2,
    spectral_norm_kwargs=None,
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
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(slope, inplace=True))
            elif activation == "elu":
                layers.append(nn.ELU(inplace=True))
            else:
                raise NotImplementedError(f"Activation '{activation}' not implemented!")

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    if spectral_norm_kwargs is not None:
        layers.append(
            spectral_norm_fc(
                nn.Linear(dim_list[-2], dim_list[-1], bias=bias), **spectral_norm_kwargs
            )
        )
    else:
        layers.append(nn.Linear(dim_list[-2], dim_list[-1], bias=bias))
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
