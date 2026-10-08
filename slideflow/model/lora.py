"""Encoder adapter utilities for MIL models."""

import torch
from torch.utils.checkpoint import checkpoint

from .extractors._lora import LoRAQKV, adapter_state_dict, apply_lora, init_lora

__all__ = ['LoRAQKV', 'init_lora', 'apply_lora', 'adapter_state_dict',
           'encode_tiles']


def encode_tiles(model, images, *, checkpoint_blocks=False):
    """Encode normalized tiles through a frozen timm ViT prefix and adapted suffix."""
    first = getattr(model, 'first_adapted', len(model.blocks))
    with torch.no_grad():
        x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(images))))
        for block in model.blocks[:first]:
            x = block(x)
    x = x.detach()
    for block in model.blocks[first:]:
        x = checkpoint(block, x, use_reentrant=False) if checkpoint_blocks else block(x)
    return model.forward_head(model.norm(x), pre_logits=True)
