import torch

from uncertainty_est.utils.utils import to_np
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.models.priornet.dpn_losses import (
    DirichletKLLoss,
    PriorNetMixedLoss,
)


class PriorNet(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        target_concentration,
        concentration,
        reverse_kl,
        alpha_fix,
        gamma,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        arch = get_arch(arch_name, arch_config)
        self.backbone = arch

        id_criterion = DirichletKLLoss(
            target_concentration=self.target_concentration,
            concentration=self.concentration,
            reverse=self.reverse_kl,
            alpha_fix=self.alpha_fix,
        )

        ood_criterion = DirichletKLLoss(
            target_concentration=1.0,
            concentration=self.concentration,
            reverse=self.reverse_kl,
            alpha_fix=self.alpha_fix,
        )

        self.criterion = PriorNetMixedLoss(
            [id_criterion, ood_criterion], mixing_params=[1.0, self.gamma]
        )

    def forward(self, x):
        logits = to_np(self.backbone(x))
        uncertanties = dirichlet_prior_network_uncertainty(
            logits, alpha_correction=self.alpha_fix
        )
        return uncertanties["differential_entropy"]

    def training_step(self, batch, batch_idx):
        (x, y), (x_ood, _) = batch

        y_hat = self.backbone(torch.cat((x, x_ood)))
        y_hat_ood = y_hat[len(x) :]
        y_hat = y_hat[: len(x)]

        loss = self.criterion((y_hat, y_hat_ood), (y, None))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y), (x_ood, _) = batch

        y_hat = self.backbone(torch.cat((x, x_ood)))
        y_hat_ood = y_hat[len(x) :]
        y_hat = y_hat[: len(x)]

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

        loss = self.criterion((y_hat, y_hat_ood), (y, None))
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.backbone(x)
        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)
        return [optim], [scheduler]

    def classify(self, x):
        logits = self.backbone(x)
        alphas = torch.exp(logits)
        if self.alpha_fix:
            alphas += 1
        return alphas / torch.sum(alphas, 0, keepdim=True)

    def get_ood_scores(self, x):
        logits = to_np(self.backbone(x))
        uncertanties = dirichlet_prior_network_uncertainty(
            logits, alpha_correction=self.alpha_fix
        )
        return uncertanties
