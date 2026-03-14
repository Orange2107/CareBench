from .registry import ModelRegistry
from .drfuse import DrFuse
from .medfuse import MedFuse
from .mmtm import MMTM
from .daft import DAFT
from .healnet import HealNetLightning
from .m3care import M3Care
from .flexmoe import FlexMoELightning
from .smil import SMIL
from .utde  import UTDE
from .umse import UMSE
from .shaspec import ShaSpec
from .lstm import LSTMModel
from .resnet import ResNetModel
from .transformer import TransformerModel
from .simple import LateFusion
from .aug import AUG
from .inforeg import InfoReg
from .crossvpt import CrossVPTLightning

def get_model(model_name: str, hparams: dict):
    """ get model class and init it with hparams"""
    model_cls = ModelRegistry.get_model_cls(model_name)
    return model_cls(hparams)

def get_model_cls(model_name: str):
    """get model class"""
    return ModelRegistry.get_model_cls(model_name)

__all__ = ['get_model', 'get_model_config_path', 
           'get_model_cls', 'DrFuse', 'MedFuse', 'MMTM', 'DAFT', 'HealNetLightning', 
            'FlexMoELightning', 'M3Care', 'SMIL', 'ShaSpec', 'UTDE', 'UMSE', 
            'LSTMModel', 'ResNetModel', 'TransformerModel', 'LateFusion', 'AUG', 'InfoReg',
            'CrossVPTLightning',]
