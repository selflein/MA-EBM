import torch
from torch import nn
from torch.distributions import Dirichlet

from uncertainty_est.models.ebm.vera import VERA
from uncertainty_est.models.priornet.dpn_losses import UnfixedDirichletKLLoss
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class VERAPriorNet(VERA):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        n_classes,
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
        alpha_fix=True,
        concentration=1.0,
        target_concentration=None,
        entropy_reg=0.0,
        reverse_kl=True,
        temperature=1.0,
        w_neg_sample_loss=0.0,
        w_neg_entropy_loss=0.0,
        **kwargs,
    ):
        super().__init__(
            arch_name,
            arch_config,
            learning_rate,
            beta1,
            beta2,
            weight_decay,
            n_classes,
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
            **kwargs,
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.clf_loss = UnfixedDirichletKLLoss(
            concentration, target_concentration, entropy_reg, reverse_kl, alpha_fix
        )
        self.temperature = temperature

    def classifier_loss(self, ld_logits, y_l, lg_logits):
        loss = self.clf_loss(ld_logits, y_l)
        self.log("train/clf_loss", loss)

        loss_ood = 0.0
        if self.w_neg_sample_loss > 0:
            loss_ood = self.w_neg_sample_loss * self.clf_loss(lg_logits)
            self.log("train/w_neg_sample_loss", loss_ood)

        loss_ood_ent = 0.0
        if self.w_neg_entropy_loss > 0:
            lg_alphas = torch.exp(lg_logits)
            if self.alpha_fix:
                lg_alphas = lg_alphas + 1
            loss_ood_ent = (
                self.w_neg_entropy_loss * -Dirichlet(lg_alphas).entropy().mean()
            )
            self.log("train/w_neg_entropy_loss", loss_ood_ent)

        return loss + loss_ood_ent + loss_ood

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)
        alphas = torch.exp(outputs[0]).reshape(-1) + self.concentration
        self.logger.experiment.add_histogram("alphas", alphas, self.current_epoch)

    def get_ood_scores(self, x):
        px, logits = self.model(x, return_logits=True)
        uncert = {}
        uncert["p(x)"] = px
        dirichlet_uncerts = dirichlet_prior_network_uncertainty(
            logits.cpu().numpy(), alpha_correction=self.alpha_fix
        )
        uncert = {**uncert, **dirichlet_uncerts}
        return uncert

    def classify(self, x):
        _, logits = self.model(x, return_logits=True)
        logits /= self.temperature
        alphas = torch.exp(logits)

        if self.alpha_fix:
            alphas += 1

        return alphas / torch.sum(alphas, 1, keepdim=True)
