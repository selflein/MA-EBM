from copy import deepcopy

import torch
from torch import nn

from uncertainty_est.models import CEBaseline


class DeepEnsemble(CEBaseline):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        num_models,
        **kwargs
    ):
        super().__init__(
            arch_name, arch_config, learning_rate, momentum, weight_decay, **kwargs
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.models = nn.ModuleList(
            [deepcopy(self.backbone) for _ in range(num_models)]
        )

    def forward(self, x):
        samples = []
        for model in self.models:
            model.to(self.device)
            samples.append(model(x).cpu())
            model.cpu()

        return torch.stack(samples, 1).var(1).mean(1)

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def classify(self, x):
        samples = []
        for model in self.models:
            model.to(self.device)
            samples.append(torch.softmax(model(x), 1).cpu())
            model.cpu()
        return torch.stack(samples).mean(0)

    def get_ood_scores(self, x):
        samples = []
        for model in self.models:
            model.to(self.device)
            samples.append(torch.softmax(model(x), 1).cpu())
            model.cpu()
        samples = torch.stack(samples, 1)

        variance = samples.var(1).mean(1)

        expected_dist = samples.mean(1)
        entropy = -(expected_dist * torch.log(expected_dist)).sum(1)

        return {"Variance": -variance, "Entropy": -entropy}
