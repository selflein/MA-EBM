from torchvision.models import vgg16

from uncertainty_est.archs.wrn import WideResNet
from uncertainty_est.archs.lipschitz_wrn import WideResNet as LipschitzWRN
from uncertainty_est.archs.fc import SynthModel
from uncertainty_est.archs.resnet import ResNetGenerator
from uncertainty_est.archs.invertible_residual_nets.net import iResNetFC
from uncertainty_est.archs.invertible_residual_nets.conv_net import iResNetConv
from uncertainty_est.archs.flows import NormalizingFlowDensity
from uncertainty_est.archs.real_nvp.real_nvp import RealNVP
from uncertainty_est.archs.seq import (
    SequenceClassifier,
    SequenceGenerativeModel,
    SequenceGenerator,
)


def get_arch(name, config_dict: dict):
    if name == "wrn":
        return WideResNet(**config_dict)
    if name == "lipschitz_wrn":
        return LipschitzWRN(**config_dict)
    elif name == "vgg16":
        return vgg16(**config_dict)
    elif name == "fc":
        return SynthModel(**config_dict)
    elif name == "resnetgenerator":
        return ResNetGenerator(**config_dict)
    elif name == "iresnet_fc":
        return iResNetFC(**config_dict)
    elif name == "iresnet_conv":
        return iResNetConv(**config_dict)
    elif name == "normalizing_flow":
        return NormalizingFlowDensity(**config_dict)
    elif name == "seq_classifier":
        return SequenceClassifier(**config_dict)
    elif name == "seq_generative_model":
        return SequenceGenerativeModel(**config_dict)
    elif name == "seq_generator":
        return SequenceGenerator(**config_dict)
    else:
        raise ValueError(f'Architecture "{name}" not implemented!')
