import torch


def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    return (y == y_hat.argmax(dim=1)).float().mean(0).item()


def get_ood_calibration(preds, reduction="mean"):
    targets = torch.ones_like(preds) / preds.size(1)
    # Use KL divergence to the uniform distribution as measure
    ood_eces = (targets * (targets.log() - (preds + 1e-8).log())).sum(-1).numpy()
    if reduction == "mean":
        return ood_eces.mean()
    else:
        return ood_eces
