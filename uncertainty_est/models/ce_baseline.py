import torch
import numpy as np
import torch.nn.functional as F

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class CEBaseline(OODDetectionModel):
    def __init__(
        self, arch_name, arch_config, learning_rate, momentum, weight_decay, **kwargs
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.backbone = get_arch(arch_name, arch_config)

    def forward(self, x):
        return self.get_ood_scores(x)["p(x)"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("val/acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test/acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)
        return [optim], [scheduler]

    def classify(self, x):
        return torch.softmax(self.backbone(x), -1)

    def get_ood_scores(self, x):
        logits = self.backbone(x).cpu()
        dir_uncert = dirichlet_prior_network_uncertainty(logits)
        dir_uncert["p(x)"] = logits.logsumexp(1)
        dir_uncert["max p(y|x)"] = logits.softmax(1).max(1).values
        return dir_uncert

    def validation_epoch_end(self, outputs):
        if hasattr(self, "ood_val_loaders"):
            avg_densities = {}
            for ds_name, loader in self.ood_val_loaders + [
                ("ID", self.val_dataloader.dataloader)
            ]:
                pxs = []
                for x, _ in loader:
                    pxs.append(
                        self.backbone(x.to(self.device)).logsumexp(-1).detach().cpu()
                    )
                avg_densities[ds_name] = torch.cat(pxs).mean().item()
            self.logger.experiment.add_scalars(
                "val/avg_densities", avg_densities, self.trainer.global_step
            )

        super().validation_epoch_end(outputs)
