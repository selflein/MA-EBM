import torch

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models import CEBaseline


class DeepEnsemble(CEBaseline):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        checkpoints,
        **kwargs
    ):
        super().__init__(
            arch_name, arch_config, learning_rate, momentum, weight_decay, **kwargs
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.models = [self.backbone]
        for ckpt in checkpoints:
            backbone = get_arch(arch_name, arch_config)
            # Remove the first module name in order to load backbone model
            strip_sd = {
                k.split(".", 1)[1]: v for k, v in torch.load(ckpt)["state_dict"].items()
            }
            backbone.load_state_dict(strip_sd)
            self.models.append(backbone)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        raise ValueError("Training not implemented")

    def configure_optimizers(self):
        raise ValueError("Training not implemented")

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
            samples.append(model(x).cpu())
            model.cpu()

        variance = torch.stack(samples, 1).var(1).mean(1)
        return {"Variance": -variance}
