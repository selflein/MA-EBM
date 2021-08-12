import torch
import numpy as np
import torch.nn.functional as F

from uncertainty_est.models.ce_baseline import CEBaseline


class EnergyFinetune(CEBaseline):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        score,
        m_in,
        m_out,
        checkpoint,
        **kwargs
    ):
        super().__init__(
            arch_name, arch_config, learning_rate, momentum, weight_decay, **kwargs
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()
        self.load_state_dict(torch.load(checkpoint)["state_dict"])

    def forward(self, x):
        return self.backbone(x).logsumexp(1)

    def finetune_loss(self, y_hat_ood, y_hat):
        if self.score == "energy":
            Ec_out = -torch.logsumexp(y_hat_ood, dim=1)
            Ec_in = -torch.logsumexp(y_hat, dim=1)
            loss = 0.1 * (
                (F.relu(Ec_in - self.m_in) ** 2).mean()
                + (F.relu(self.m_out - Ec_out) ** 2).mean()
            )
            self.log("train/margin_loss", loss, prog_bar=True)
        elif self.score == "OE":
            loss = 0.5 * -(y_hat_ood.mean(1) - torch.logsumexp(y_hat_ood, dim=1)).mean()
        return loss

    def training_step(self, batch, batch_idx):
        (x, y), (x_ood, _) = batch

        y_hat = self(torch.cat((x, x_ood)))
        y_hat_ood = y_hat[len(x) :]
        y_hat = y_hat[: len(x)]
        loss = F.cross_entropy(y_hat, y)
        loss += self.finetune_loss(y_hat_ood, y_hat)

        self.log("train/ce_loss", loss, prog_bar=True)

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y), (x_ood, y_ood) = batch
        y_hat = self.backbone(x)
        y_hat_ood = self.backbone(x_ood)

        loss = F.cross_entropy(y_hat, y)
        loss += self.finetune_loss(y_hat_ood, y_hat)
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

        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lr_lambda=lambda step: cosine_annealing(
                step,
                self.trainer.max_epochs * len(self.train_dataloader.dataloader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / self.learning_rate,
            ),
        )
        return [optim], [scheduler]

    def get_ood_scores(self, x):
        logits = self.backbone(x).cpu()
        ood_scores = {}
        ood_scores["p(x)"] = logits.logsumexp(1)
        ood_scores["max p(y|x)"] = logits.softmax(1).max(1).values
        return ood_scores
