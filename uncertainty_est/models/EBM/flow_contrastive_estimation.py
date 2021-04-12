from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import distributions
import matplotlib.pyplot as plt

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.utils.utils import (
    to_np,
    estimate_normalizing_constant,
    sum_except_batch,
)


class FlowContrastiveEstimation(OODDetectionModel):
    """Implementation of Noise Contrastive Estimation http://proceedings.mlr.press/v9/gutmann10a.html"""

    def __init__(
        self,
        arch_name,
        arch_config,
        flow_arch_name,
        flow_arch_config,
        learning_rate,
        momentum,
        weight_decay,
        flow_learning_rate,
        is_toy_dataset=False,
        toy_dataset_dim=2,
        test_ood_dataloaders=[],
    ):
        super().__init__(test_ood_dataloaders)
        self.automatic_optimization = False
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.model = get_arch(arch_name, arch_config)
        self.noise_dist = get_arch(flow_arch_name, flow_arch_config)

    def forward(self, x):
        return self.model(x)

    def compute_ebm_loss(self, batch, return_outputs=False):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        optim_ebm, optim_flow = self.optimizers()
        x, _ = batch

        optim_ebm.zero_grad()
        optim_flow.zero_grad()

        noise, flow_log_p_noise = self.noise_dist.sample(len(x))
        flow_log_p_x = sum_except_batch(self.noise_dist.log_prob(x))
        flow_log_p = torch.cat((flow_log_p_x, flow_log_p_noise))

        inp = torch.cat((x, noise))
        logits = self.model(inp)
        log_p_model = logits.logsumexp(-1)

        out = log_p_model - flow_log_p
        targets = torch.cat((torch.ones(len(x)), torch.zeros(len(x)))).to(self.device)
        loss = F.binary_cross_entropy_with_logits(out, targets)

        acc = ((torch.sigmoid(out) > 0.5) == targets).float().mean()
        self.log("acc", acc, prog_bar=True)

        if acc < 0.5:
            self.manual_backward(loss)
            optim_ebm.step()
            self.log("train/loss", loss)
        else:
            self.manual_backward(-loss)
            optim_flow.step()
            self.log("train/flow_loss", -loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return

    def validation_epoch_end(self, outputs):
        if self.is_toy_dataset and self.toy_dataset_dim == 2:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1)
            p_xy = torch.exp(self(data.to(self.device)))
            px = to_np(p_xy.sum(1))

            x, y = to_np(x), to_np(y)
            for i in range(p_xy.shape[1]):
                fig, ax = plt.subplots()
                mesh = ax.pcolormesh(x, y, to_np(p_xy[:, i]).reshape(*x.shape))
                fig.colorbar(mesh)
                self.logger.experiment.add_figure(
                    f"dist/p(x,y={i})", fig, self.current_epoch
                )
                plt.close()

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x, y, px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("dist/p(x)", fig, self.current_epoch)
            plt.close()

    def test_step(self, batch, batch_idx):
        self.to(torch.float32)
        x, y = batch
        y_hat = self.model(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)
        return y_hat

    def test_epoch_end(self, logits):
        if self.is_toy_dataset:
            # Estimate normalizing constant Z by numerical integration
            log_Z = torch.log(
                estimate_normalizing_constant(
                    lambda x: self(x).exp().sum(1),
                    device=self.device,
                    dimensions=self.toy_dataset_dim,
                    dtype=torch.float32,
                )
            )
            self.log("log_Z", log_Z)

            logits = torch.cat(logits, 0)
            log_px = logits.logsumexp(1) - log_Z
            self.log("log_likelihood", log_px.mean())

        super().test_epoch_end()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)

        optim_flow = torch.optim.AdamW(
            self.noise_dist.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.flow_learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler_flow = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)

        return [optim, optim_flow], [scheduler, scheduler_flow]

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        scores = []
        for x, y in tqdm(loader):
            x = x.to(self.device)
            score = self.model(x).cpu()
            scores.append(score)

        uncert = {}
        uncert["p(x)"] = torch.cat(scores).cpu().numpy()
        return uncert
