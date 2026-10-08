"""Opt-in nnMIL, LoRA and Mettle registration through Slideflow's public hooks."""

from slideflow.mil import register_model
from slideflow.model.extractors._registry import register_torch
from ._extension_config import LoRAConfig, LoRAModelConfig, NNMILModelConfig
from ._extension_data import resolve_tile_bags


def mil_config(model, trainer='fastai', **kwargs):
    """Create an opt-in LoRA config accepting TFRecord directories as bags."""
    from slideflow.mil import mil_config as base_config

    if model == 'lora' and trainer == 'fastai':
        return LoRAConfig(model=model, **kwargs)
    return base_config(model, trainer=trainer, **kwargs)


def load_mil_config(path, strict=False):
    """Restore an opt-in config, including tile directory discovery for LoRA."""
    import slideflow as sf

    config = sf.mil.load_mil_config(path, strict=strict)
    if config.model_config.model == 'lora':
        return mil_config(trainer=config.tag, **config.to_dict(), validate=strict)
    return config


def build_fastai_learner(config, train_dataset, val_dataset, outcomes, bags, **kwargs):
    """Build the ordinary FastAI learner from feature or TFRecord tile bags."""
    import slideflow as sf

    if config.model_config.model == 'lora':
        bags = resolve_tile_bags(bags)
    return sf.mil.build_fastai_learner(config, train_dataset, val_dataset, outcomes, bags, **kwargs)


def eval_mil(weights, dataset, outcomes, bags, config=None, **kwargs):
    """Evaluate saved LoRA weights directly from tile bag directories."""
    import slideflow as sf

    if config is None:
        config = load_mil_config(weights)
    if config.model_config.model == 'lora':
        bags = resolve_tile_bags(bags)
    return sf.mil.eval_mil(weights, dataset, outcomes, bags, config=config, **kwargs)


def predict_mil(model, dataset, outcomes, bags, *, config=None, **kwargs):
    """Predict from tile directories using the standard MIL prediction path."""
    import slideflow as sf

    if isinstance(model, str):
        model, config = sf.mil.load_model_weights(model, config)
    if config is not None and config.model_config.model == 'lora':
        bags = resolve_tile_bags(bags)
    return sf.mil.predict_mil(model, dataset, outcomes, bags, config=config, **kwargs)


@register_model('nnmil', config=NNMILModelConfig)
def nnmil():
    from .models.nnmil import NNMIL
    return NNMIL


@register_model('lora', config=LoRAModelConfig)
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
