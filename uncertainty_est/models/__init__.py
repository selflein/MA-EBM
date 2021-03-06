from os import POSIX_FADV_SEQUENTIAL
import yaml
from pathlib import Path

from uncertainty_est.models.ebm.hdge import HDGEModel
from uncertainty_est.models.ebm.mcmc import MCMC
from uncertainty_est.models.ebm.discrete_mcmc import DiscreteMCMC
from uncertainty_est.models.ce_baseline import CEBaseline
from uncertainty_est.models.energy_finetuning import EnergyFinetune
from uncertainty_est.models.priornet.priornet import PriorNet
from uncertainty_est.models.ebm.vera import VERA
from uncertainty_est.models.ebm.vera_priornet import VERAPriorNet
from uncertainty_est.models.ebm.vera_posteriornet import VERAPosteriorNet
from uncertainty_est.models.normalizing_flow.norm_flow import NormalizingFlow
from uncertainty_est.models.normalizing_flow.approx_flow import ApproxNormalizingFlow
from uncertainty_est.models.ebm.conditional_nce import PerSampleNCE
from uncertainty_est.models.normalizing_flow.iresnet import IResNetFlow
from .normalizing_flow.image_flows import RealNVPModel, GlowModel
from .vae import VAE
from .ebm.nce import NoiseContrastiveEstimation
from .ebm.flow_contrastive_estimation import FlowContrastiveEstimation
from .ebm.ssm import SSM
from .autoencoder import Autoencoder
from .mc_dropout import MCDropout
from .deep_ensemble import DeepEnsemble
from .ebm.mcmc_priornet import MCMCPriorNet
from .ebm.mcmc_entropy import EntropyMCMC
from .postnet import PosteriorNetwork


MODELS = {
    "HDGE": HDGEModel,
    "JEM": MCMC,
    "DiscreteMCMC": DiscreteMCMC,
    "CEBaseline": CEBaseline,
    "EnergyOOD": EnergyFinetune,
    "PriorNet": PriorNet,
    "VERA": VERA,
    "VERAPriorNet": VERAPriorNet,
    "VERAPosteriorNet": VERAPosteriorNet,
    "NormalizingFlow": NormalizingFlow,
    "ApproxNormalizingFlow": ApproxNormalizingFlow,
    "PerSampleNCE": PerSampleNCE,
    "IResNet": IResNetFlow,
    "RealNVP": RealNVPModel,
    "Glow": GlowModel,
    "VAE": VAE,
    "NCE": NoiseContrastiveEstimation,
    "FlowCE": FlowContrastiveEstimation,
    "SSM": SSM,
    "Autoencoder": Autoencoder,
    "MCDropout": MCDropout,
    "DeepEnsemble": DeepEnsemble,
    "MCMCPriorNet": MCMCPriorNet,
    "EntropyMCMC": EntropyMCMC,
    "PostNet": PosteriorNetwork,
}


def resolve_model_checkpoint(model_folder: Path, last=False):
    ckpts = [file for file in model_folder.iterdir() if file.suffix == ".ckpt"]
    last_ckpt = [ckpt for ckpt in ckpts if ckpt.stem == "last"]
    best_ckpt = sorted([ckpt for ckpt in ckpts if ckpt.stem != "last"])

    if last:
        ckpts = best_ckpt + last_ckpt
    else:
        ckpts = last_ckpt + best_ckpt
    assert len(ckpts) > 0

    return ckpts[-1]


def load_model(model_folder: Path, last=False, *args, **kwargs):
    checkpoint_path = resolve_model_checkpoint(model_folder, last)
    return load_checkpoint(checkpoint_path, *args, **kwargs)


def load_checkpoint(checkpoint_path: Path, *args, **kwargs):
    with (checkpoint_path.parent / "config.yaml").open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    try:
        with (checkpoint_path.parent / "hparams.yaml").open("r") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
        if "checkpoint" in hparams:
            hparams.pop("checkpoint")
    except:
        hparams = {}

    kwargs = {**kwargs, **hparams}

    model = MODELS[config["model_name"]].load_from_checkpoint(
        checkpoint_path, *args, **kwargs
    )
    return model, config
