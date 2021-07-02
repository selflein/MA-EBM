import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from uncertainty_est.models.ebm.mcmc import MCMC
from uncertainty_est.models.priornet.dpn_losses import UnfixedDirichletKLLoss
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class MCMCPriorNet(MCMC):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        buffer_size,
        n_classes,
        data_shape,
        smoothing,
        pyxce,
        pxsgld,
        pxysgld,
        class_cond_p_x_sample,
        sgld_batch_size,
        sgld_lr,
        sgld_std,
        reinit_freq,
        sgld_steps=20,
        entropy_reg_weight=0.0,
        warmup_steps=2500,
        lr_step_size=50,
        is_toy_dataset=False,
        alpha_fix=True,
        concentration=1.0,
        target_concentration=None,
        reverse_kl=True,
        w_neg_sample_loss=0.0,
        w_neg_entropy_loss=0.0,
        **kwargs
    ):
        super().__init__(
            arch_name=arch_name,
            arch_config=arch_config,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            buffer_size=buffer_size,
            n_classes=n_classes,
            data_shape=data_shape,
            smoothing=smoothing,
            pyxce=pyxce,
            pxsgld=pxsgld,
            pxysgld=pxysgld,
            class_cond_p_x_sample=class_cond_p_x_sample,
            sgld_batch_size=sgld_batch_size,
            sgld_lr=sgld_lr,
            sgld_std=sgld_std,
            reinit_freq=reinit_freq,
            sgld_steps=sgld_steps,
            entropy_reg_weight=entropy_reg_weight,
            warmup_steps=warmup_steps,
            lr_step_size=lr_step_size,
            is_toy_dataset=is_toy_dataset,
            **kwargs
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.clf_loss = UnfixedDirichletKLLoss(
            concentration,
            target_concentration,
            entropy_reg_weight,
            reverse_kl,
            alpha_fix,
        )

    def classifier_loss(self, ld_logits, y_l, lg_logits):
        alpha = torch.exp(ld_logits)  # / self.p_y.unsqueeze(0).to(self.device)
        # Multiply by class counts for Bayesian update

        if self.alpha_fix:
            alpha = alpha + 1

        soft_output = F.one_hot(y_l, self.n_classes)
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.n_classes)
        UCE_loss = torch.mean(
            soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))
        )
        UCE_loss = (
            UCE_loss + self.entropy_reg_weight * -Dirichlet(alpha).entropy().mean()
        )
        self.log("train/clf_loss", UCE_loss)

        return UCE_loss

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
        alphas = torch.exp(logits)
        if self.alpha_fix:
            alphas += 1

        return alphas / torch.sum(alphas, 1, keepdim=True)
