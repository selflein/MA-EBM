from itertools import islice
from os import path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from uncertainty_eval.vis import plot_score_hist
from uncertainty_eval.metrics.brier import brier_score, brier_decomposition
from uncertainty_eval.metrics.calibration_error import classification_calibration
from uncertainty_eval.vis import draw_reliability_graph, plot_score_hist

from uncertainty_est.utils.utils import to_np
from uncertainty_est.utils.metrics import accuracy
from uncertainty_est.data.dataloaders import get_dataloader


class OODDetectionModel(pl.LightningModule):
    def __init__(
        self, ood_val_datasets=None, is_toy_dataset=False, data_shape=None, **kwargs
    ):
        super().__init__()
        self.ood_val_datasets = ood_val_datasets
        self.is_toy_dataset = is_toy_dataset
        self.data_shape = data_shape
        self.test_ood_dataloaders = []

    def eval_ood(
        self, id_loader, ood_loaders: List[Tuple[str, Any]], num=10_000
    ) -> Dict[str, float]:
        self.eval()

        if num > 0:
            max_batches = (num // id_loader.batch_size) + 1
        else:
            assert num == -1
            max_batches = None

        ood_metrics = {}

        # Compute ID OOD scores
        id_scores_dict = self.ood_detect(islice(id_loader, max_batches))

        # Compute OOD detection metrics
        for dataset_name, loader in ood_loaders:
            try:
                ood_scores_dict = self.ood_detect(islice(loader, max_batches))
            except Exception as e:
                print(e)
                continue

            for score_name, id_scores in id_scores_dict.items():
                try:
                    ood_scores = ood_scores_dict[score_name]

                    length = min(len(ood_scores), len(id_scores))
                    ood = ood_scores[:length]
                    id_ = id_scores[:length]

                    if self.logger is not None and self.logger.log_dir is not None:
                        ax = plot_score_hist(
                            id_,
                            ood,
                            title="",
                        )
                        ax.figure.savefig(
                            path.join(
                                self.logger.log_dir, f"{dataset_name}_{score_name}.png"
                            )
                        )
                        plt.close()

                    preds = np.concatenate([ood, id_])

                    labels = np.concatenate([np.zeros_like(ood), np.ones_like(id_)])
                    ood_metrics[(dataset_name, score_name, "AUROC")] = (
                        roc_auc_score(labels, preds) * 100.0
                    )
                    ood_metrics[(dataset_name, score_name, "AUPR")] = (
                        average_precision_score(labels, preds) * 100.0
                    )

                    labels = np.concatenate([np.ones_like(ood), np.zeros_like(id_)])
                    ood_metrics[(dataset_name, score_name, "AUROC")] = (
                        roc_auc_score(labels, -preds) * 100.0
                    )
                    ood_metrics[(dataset_name, score_name, "AUPR")] = (
                        average_precision_score(labels, -preds) * 100.0
                    )
                except Exception as e:
                    print(e)
        return ood_metrics

    def eval_classifier(self, loader, num=10_000):
        self.eval()

        if num > 0:
            max_batches = (num // loader.batch_size) + 1
        else:
            assert num == -1
            max_batches = None

        try:
            y, probs = self.get_gt_preds(islice(loader, max_batches))
            y, probs = y[:num], probs[:num]
        except NotImplementedError:
            print("Model does not support classification.")
            return {}

        try:
            # Compute accuracy
            acc = accuracy(y, probs)

            # Compute calibration
            y_np, probs_np = to_np(y), to_np(probs)
            ece, mce = classification_calibration(y_np, probs_np)
            brier = brier_score(y_np, probs_np)
            uncertainty, resolution, reliability = brier_decomposition(y_np, probs_np)

            if self.logger is not None and self.logger.log_dir is not None:
                fig, ax = plt.subplots(figsize=(10, 10))
                draw_reliability_graph(y_np, probs_np, 10, ax=ax)
                fig.savefig(self.logger.log_dir / "calibration.png", dpi=200)
        except:
            return {}

        return {
            "Accuracy": acc * 100,
            "ECE": ece * 100.0,
            "MCE": mce * 100.0,
            "Brier": brier * 100,
            "Brier uncertainty": uncertainty * 100,
            "Brier resolution": resolution * 100,
            "Brier reliability": reliability * 100,
            "Brier (via decomposition)": (reliability - resolution + uncertainty) * 100,
        }

    def setup(self, mode):
        if mode == "fit" and self.ood_val_datasets:
            batch_size = self.val_dataloader.dataloader.batch_size

            self.ood_val_loaders = []
            for ood_ds_name in self.ood_val_datasets:
                ood_loader = get_dataloader(
                    ood_ds_name,
                    "val",
                    batch_size=batch_size,
                    data_shape=self.data_shape,
                )
                self.ood_val_loaders.append((ood_ds_name, ood_loader))

    def validation_epoch_end(self, outputs):
        if self.is_toy_dataset:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1).to(self.device)
            px = to_np(torch.exp(self(data)))

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x, y, px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("dist/p(x)", fig, self.current_epoch)
            plt.close()

        if hasattr(self, "ood_val_loaders"):
            ood_metrics = self.eval_ood(
                self.val_dataloader.dataloader, self.ood_val_loaders
            )
            self.logger.experiment.add_scalars(
                "val/all_ood",
                {", ".join(k): v for k, v in ood_metrics.items()},
                self.trainer.global_step,
            )

            avg_over_dataset_results = defaultdict(list)
            for k, v in ood_metrics.items():
                avg_over_dataset_results[", ".join(k[1:])].append(v)

            k, v = next(
                iter(
                    {k: np.mean(v) for k, v in avg_over_dataset_results.items()}.items()
                )
            )
            self.log(f"val/ood", v)

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
        if (
            optimizer is not None
            and hasattr(self, "warmup_steps")
            and self.trainer.global_step < self.warmup_steps
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / float(self.warmup_steps)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

        optimizer.step(closure=optimizer_closure)

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)

        scores = defaultdict(list)
        for x, _ in tqdm(loader, miniters=100):
            if not isinstance(x, torch.Tensor):
                x, _ = x
            x = x.to(self.device)
            out = self.get_ood_scores(x)
            for k, v in out.items():
                scores[k].append(to_np(v))

        scores = {k: np.concatenate(v) for k, v in scores.items()}
        return scores

    def get_gt_preds(self, loader):
        self.eval()
        torch.set_grad_enabled(False)

        gt, preds = [], []
        for x, y in tqdm(loader, miniters=100):
            x = x.to(self.device)
            y_hat = self.classify(x).cpu()
            gt.append(y)
            preds.append(y_hat)
        return torch.cat(gt), torch.cat(preds)

    def get_ood_scores(self, x) -> Dict[str, torch.tensor]:
        raise NotImplementedError

    def classify(self, x) -> torch.tensor:
        raise NotImplementedError
