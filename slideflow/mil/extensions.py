"""Opt-in nnMIL, LoRA and Mettle registration through Slideflow's public hooks."""

from slideflow.mil import register_model
from slideflow.model.extractors._registry import register_torch
from ._extension_config import ExtensionModelConfig, NNMILModelConfig


@register_model('nnmil', config=NNMILModelConfig)
def nnmil():
    from .models.nnmil import NNMIL
    return NNMIL


@register_model('lora', config=ExtensionModelConfig)
def lora():
    from .models.lora import LoRA
    return LoRA


@register_torch('mettle')
def mettle(**kwargs):
    from slideflow.model.extractors.mettle import MettleFeatures
    return MettleFeatures(**kwargs)


@register_torch('hoptimus0_lora')
def hoptimus0_lora(**kwargs):
    from slideflow.model.extractors.hoptimus0_lora import Hoptimus0LoRAFeatures
    return Hoptimus0LoRAFeatures(**kwargs)
