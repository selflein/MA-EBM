import torch


def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    return (y == y_hat.argmax(dim=1)).float().mean(0).item()


def get_ood_calibration(preds, reduction="mean"):
    targets = torch.ones_like(preds) / preds.size(1)
    ood_eces = (preds - targets).abs().max(1).values.numpy()
    if reduction == "mean":
        return ood_eces.mean()
    else:
        return ood_eces
