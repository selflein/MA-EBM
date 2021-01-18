from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as tf
from torch import distributions
import matplotlib.pyplot as plt

from uncertainty_est.utils.utils import to_np
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.JEM.model import F, ConditionalF
from uncertainty_est.models.JEM.vera_utils import (
    VERAGenerator,
    VERAHMCGenerator,
    set_bn_to_eval,
    set_bn_to_train,
)
from uncertainty_est.models.JEM.jem import JEM
from uncertainty_est.models.JEM.utils import (
    KHotCrossEntropyLoss,
    smooth_one_hot,
    init_random,
)
from uncertainty_est.models.priornet.dpn_losses import dirichlet_kl_divergence


class VERAPriorNet(pl.LightningModule):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        n_classes,
        uncond,
        gen_learning_rate,
        ebm_iters,
        generator_iters,
        entropy_weight,
        generator_type,
        generator_arch_name,
        generator_arch_config,
        generator_config,
        min_sigma,
        max_sigma,
        p_control,
        n_control,
        pg_control,
        clf_ent_weight,
        ebm_type,
        clf_weight,
        warmup_steps,
        no_g_batch_norm,
        batch_size,
        lr_decay,
        lr_decay_epochs,
        vis_every=-1,
        alpha_fix=True,
        concentration=1.0,
        target_concentration=None,
        kl_weight=1.0,
        entropy_reg=0.0,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()
        self.automatic_optimization = False

        arch = get_arch(arch_name, arch_config)
        self.model = (
            F(arch, n_classes) if self.uncond else ConditionalF(arch, n_classes)
        )

        g = get_arch(generator_arch_name, generator_arch_config)
        if generator_type == "verahmc":
            self.generator = VERAHMCGenerator(g, **generator_config)
        elif generator_type == "vera":
            self.generator = VERAGenerator(g, **generator_config)
        else:
            raise NotImplementedError(f"Generator '{generator_type}' not implemented!")

    def forward(self, x):
        return self.model.classify(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_e, opt_g = self.optimizers()
        (x_l, y_l), (x_d, _) = batch

        x_l.requires_grad_()
        x_d.requires_grad_()

        # sample from q(x, h)
        x_g, h_g = self.generator.sample(x_l.size(0), requires_grad=True)

        # ebm (contrastive divergence) objective
        if batch_idx % self.ebm_iters == 0:
            ebm_loss = self.ebm_step(x_d, x_l, x_g, y_l)

            self.log("ebm_loss", ebm_loss, prog_bar=True)

            opt_e.zero_grad()
            self.manual_backward(ebm_loss, opt_e)
            opt_e.step()

        # gen obj
        if batch_idx % self.generator_iters == 0:
            gen_loss = self.generator_step(x_g, h_g)

            self.log("gen_loss", gen_loss, prog_bar=True)

            opt_g.zero_grad()
            self.manual_backward(gen_loss, opt_g)
            opt_g.step()

        # clamp sigma to (.01, max_sigma) for generators
        if self.generator_type in ["verahmc", "vera"]:
            self.generator.clamp_sigma(self.max_sigma, sigma_min=self.min_sigma)

    def ebm_step(self, x_d, x_l, x_g, y_l):
        x_g_detach = x_g.detach().requires_grad_()

        if self.no_g_batch_norm:
            self.model.apply(set_bn_to_eval)
            lg_detach = self.model(x_g_detach).squeeze()
            self.model.apply(set_bn_to_train)
        else:
            lg_detach = self.model(x_g_detach).squeeze()

        loss = torch.tensor(0.0).to(self.device)
        unsup_ent = torch.tensor(0.0)
        if self.ebm_type == "ssl":
            ld, unsup_logits = self.model(x_d, return_logits=True)
            _, ld_logits = self.model(x_l, return_logits=True)
            unsup_ent = distributions.Categorical(logits=unsup_logits).entropy()
        elif self.ebm_type == "jem":
            ld, ld_logits = self.model(x_l, return_logits=True)
        elif self.ebm_type == "p_x":
            ld, ld_logits = self.model(x_l).squeeze(), torch.tensor(0.0).to(self.device)
        elif self.ebm_type == "priornet":
            ld, ld_logits = self.model(x_l, return_logits=True)
            alphas = torch.exp(ld_logits)
            if self.alpha_fix:
                alphas = alphas + 1

            if self.target_concentration is None:
                target_concentration = torch.exp(self.model(x_l)) + self.concentration
            else:
                target_concentration = (
                    torch.empty(len(alphas))
                    .fill_(self.target_concentration)
                    .to(self.device)
                )

            target_alphas = torch.empty_like(alphas).fill_(self.concentration)
            target_alphas[torch.arange(len(y_l)), y_l] = target_concentration
            kl_term = dirichlet_kl_divergence(target_alphas, alphas)
            loss += self.kl_weight * kl_term.mean()
            loss += (
                self.entropy_reg
                * -torch.distributions.Dirichlet(alphas).entropy().mean()
            )
        else:
            raise NotImplementedError(f"EBM type '{self.ebm_type}' not implemented!")

        grad_ld = (
            torch.autograd.grad(ld.sum(), x_l, create_graph=True)[0]
            .flatten(start_dim=1)
            .norm(2, 1)
        )

        logp_obj = (ld - lg_detach).mean()
        loss += (
            -logp_obj
            + self.p_control * (ld ** 2).mean()
            + self.n_control * (lg_detach ** 2).mean()
            + self.pg_control * (grad_ld ** 2.0 / 2.0).mean()
            + self.clf_ent_weight * unsup_ent.mean()
        )

        if self.clf_weight > 0:
            loss += self.clf_weight * torch.nn.CrossEntropyLoss()(ld_logits, y_l)

        return loss

    def generator_step(self, x_g, h_g):
        lg = self.model(x_g).squeeze()
        grad = torch.autograd.grad(lg.sum(), x_g, retain_graph=True)[0]
        ebm_gn = grad.norm(2, 1).mean()

        if self.entropy_weight != 0.0:
            entropy_obj, ent_gn = self.generator.entropy_obj(x_g, h_g)

        logq_obj = lg.mean() + self.entropy_weight * entropy_obj
        return -logq_obj

    def validation_step(self, batch, batch_idx):
        (x_l, y_l), (x_d, _) = batch
        logits = self(x_l)

        log_px = self.model(x_l).mean()
        self.log("val_loss", -log_px)

        # Performing density estimation only
        if logits.shape[1] < 2:
            return

        acc = (y_l == logits.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

    def validation_epoch_end(self, training_step_outputs):
        if self.vis_every > 0 and self.current_epoch % self.vis_every == 0:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1)

            px = to_np(torch.exp(self.model(data.to(self.device))))

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(to_np(x), to_np(y), px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("p(x)", fig, self.current_epoch)
            plt.close()

    def test_step(self, batch, batch_idx):
        (x, y), (_, _) = batch
        y_hat = self(x)

        # Performing density estimation only
        if y_hat.shape[1] < 2:
            return
        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            betas=(self.beta1, self.beta2),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        gen_optim = torch.optim.AdamW(
            self.generator.parameters(),
            betas=(self.beta1, self.beta2),
            lr=self.gen_learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, gamma=self.lr_decay, milestones=self.lr_decay_epochs
        )
        gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            gen_optim, gamma=self.lr_decay, milestones=self.lr_decay_epochs
        )
        return [optim, gen_optim], [scheduler, gen_scheduler]

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = None,
        optimizer_closure=None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
        **kwargs,
    ):
        # learning rate warm-up
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / float(self.warmup_steps)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

        optimizer.step(closure=optimizer_closure)

    def get_gt_preds(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        gt, preds = [], []
        for x, y in tqdm(loader):
            x = x.to(self.device)
            y_hat = self(x).cpu()
            gt.append(y)
            preds.append(y_hat)
        return torch.cat(gt), torch.cat(preds)

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