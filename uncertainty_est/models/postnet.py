"""
Adapted from https://github.com/sharpenb/Posterior-Network/blob/main/src/posterior_networks/PosteriorNetwork.py
"""

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

from uncertainty_est.archs.fc import SynthModel
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.archs.flows import NormalizingFlowDensity
from uncertainty_est.models.ood_detection_model import OODDetectionModel


__budget_functions__ = {
    "one": lambda N: torch.ones_like(N),
    "log": lambda N: torch.log(N + 1.0),
    "id": lambda N: N,
    "id_normalized": lambda N: N / N.sum(),
    "exp": lambda N: torch.exp(N),
    "parametrized": lambda N: torch.ones_like(N),
}


class PosteriorNetwork(OODDetectionModel):
    def __init__(
        self,
        output_dim,  # Output dimension. int
        learning_rate=1e-3,
        hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
        latent_dim=10,  # Latent dimension. int
        arch_name="wrn",  # Encoder architecture name. int
        arch_config={},
        no_density=False,  # Use density estimation or not. boolean
        density_type="radial_flow",  # Density type. string
        n_density=8,  # Number of density components. int
        budget_function="id",  # Budget function name applied on class count. name
        loss="UCE",  # Loss name. string
        regr=1e-5,
        N=[
            1,
        ],
        **kwargs
    ):  # Regularization factor in Bayesian loss. float
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if not hasattr(self, "N"):
            self.N = torch.tensor(N).to(self.device)

        # Architecture parameters
        self.output_dim, self.hidden_dims, self.latent_dim = (
            output_dim,
            hidden_dims,
            latent_dim,
        )
        self.no_density, self.density_type, self.n_density = (
            no_density,
            density_type,
            n_density,
        )

        # Training parameters
        self.loss, self.regr = loss, regr
        self.learning_rate = learning_rate
        self.budget_function = budget_function

        # Encoder -- Feature selection
        self.sequential = get_arch(arch_name, arch_config)

        self.batch_norm = nn.BatchNorm1d(num_features=self.latent_dim)

        self.linear_classifier = SynthModel(
            inp_dim=self.latent_dim,
            num_classes=self.output_dim,
            hidden_dims=[self.hidden_dims[-1]],
        )

        # Normalizing Flow -- Normalized density on latent space
        if self.density_type == "planar_flow":
            self.density_estimation = nn.ModuleList(
                [
                    NormalizingFlowDensity(
                        dim=self.latent_dim,
                        flow_length=n_density,
                        flow_type=self.density_type,
                    )
                    for c in range(self.output_dim)
                ]
            )
        elif self.density_type == "radial_flow":
            self.density_estimation = nn.ModuleList(
                [
                    NormalizingFlowDensity(
                        dim=self.latent_dim,
                        flow_length=n_density,
                        flow_type=self.density_type,
                    )
                    for c in range(self.output_dim)
                ]
            )
        elif self.density_type == "iaf_flow":
            self.density_estimation = nn.ModuleList(
                [
                    NormalizingFlowDensity(
                        dim=self.latent_dim,
                        flow_length=n_density,
                        flow_type=self.density_type,
                    )
                    for c in range(self.output_dim)
                ]
            )
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=-1)

    def setup(self, phase):
        if phase == "fit":
            train_dataset = self.train_dataloader.dataloader.dataset

            class_counts = torch.zeros(self.output_dim)
            for _, y in train_dataset:
                class_counts[y] += 1

            if self.budget_function in __budget_functions__:
                self.N = nn.Parameter(
                    __budget_functions__[self.budget_function](class_counts)
                ).to(self.device)
            else:
                raise NotImplementedError

    def forward(self, input, return_output="soft"):
        batch_size = input.size(0)

        if self.N.device != input.device:
            self.N = self.N.to(input.device)

        if self.budget_function == "parametrized":
            N = self.N / self.N.sum()
        else:
            N = self.N

        # Forward
        zk = self.sequential(input)
        if self.no_density:  # Ablated model without density estimation
            logits = self.linear_classifier(zk)
            alpha = torch.exp(logits)
            soft_output_pred = self.softmax(logits)
        else:  # Full model with density estimation
            zk = self.batch_norm(zk)
            log_q_zk = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)
            alpha = torch.zeros((batch_size, self.output_dim)).to(zk.device.type)

            if isinstance(self.density_estimation, nn.ModuleList):
                for c in range(self.output_dim):
                    log_p = self.density_estimation[c].log_prob(zk)
                    log_q_zk[:, c] = log_p
                    alpha[:, c] = 1.0 + (N[c] * torch.exp(log_q_zk[:, c]))
            else:
                log_q_zk = self.density_estimation.log_prob(zk)
                alpha = 1.0 + (N[:, None] * torch.exp(log_q_zk)).permute(1, 0)

            pass

            soft_output_pred = F.normalize(alpha, p=1)

        if return_output == "soft":
            return soft_output_pred
        elif return_output == "alpha":
            return alpha
        elif return_output == "latent":
            return zk
        else:
            raise AssertionError

    def CE_loss(self, soft_output_pred, soft_output):
        return -torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))

    def UCE_loss(self, alpha, soft_output):
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
        entropy_reg = Dirichlet(alpha).entropy()
        UCE_loss = torch.sum(
            soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))
        ) - self.regr * torch.sum(entropy_reg)

        return UCE_loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        soft_output = F.one_hot(y, self.output_dim)
        if self.loss == "CE":
            soft_output_pred = self.forward(x, return_output="soft")
            loss = self.CE_loss(soft_output_pred, soft_output)
        elif self.loss == "UCE":
            alpha = self.forward(x, return_output="alpha")
            loss = self.UCE_loss(alpha, soft_output)
        else:
            raise NotImplementedError

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        soft_output_pred = self.forward(x, return_output="soft")
        acc = (soft_output_pred.argmax(-1) == y).float().mean()
        self.log("val/acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def classify(self, x):
        return self.forward(x, return_output="soft")

    def get_ood_scores(self, x):
        alpha = self.forward(x, return_output="soft")
        soft_output_pred = F.normalize(alpha, p=1)
        return {
            "max p(y|x)": torch.max(soft_output_pred, -1).values,
            "sum alpha": torch.sum(alpha, 1),
        }
