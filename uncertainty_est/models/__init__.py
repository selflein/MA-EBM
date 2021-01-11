from uncertainty_est.models.JEM.hdge import HDGEModel
from uncertainty_est.models.JEM.jem import JEM
from uncertainty_est.models.JEM.jem_priornet import JEMPriorNet
from uncertainty_est.models.JEM.hdge_priornet import HDGEPriorNetModel
from uncertainty_est.models.ce_baseline import CEBaseline
from uncertainty_est.models.energy_finetuning import EnergyFinetune
from uncertainty_est.models.priornet.priornet import PriorNet
from uncertainty_est.models.JEM.vera import VERA

MODELS = {
    "HDGE": HDGEModel,
    "JEM": JEM,
    "CEBaseline": CEBaseline,
    "EnergyOOD": EnergyFinetune,
    "PriorNet": PriorNet,
    "JEMPriorNet": JEMPriorNet,
    "HDGEPriorNet": HDGEPriorNetModel,
    "VERA": VERA,
}
