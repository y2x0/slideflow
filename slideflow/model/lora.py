"""Train encoder adapters with a user-supplied prediction model and loss."""

import torch
from torch.utils.checkpoint import checkpoint

from .extractors._lora import LoRAQKV, adapter_state_dict, apply_lora, init_lora

__all__ = ['LoRAQKV', 'init_lora', 'apply_lora', 'adapter_state_dict',
           'encode_tiles', 'build_lora_learner', 'train_lora', 'predict_lora']


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


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    value = torch.as_tensor(value, device=device)
    return value.float() if value.is_floating_point() else value


def _cpu(value):
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


def _prepare(extractor, batch, device):
    if isinstance(batch, dict):
        tiles, target, lengths = batch['tiles'], batch['targets'], batch.get('lengths')
    elif len(batch) in (2, 3):
        tiles, target = batch[:2]
        lengths = batch[2] if len(batch) == 3 else None
    else:
        raise ValueError('batches need tiles, targets and optional bag lengths')
    tiles = torch.as_tensor(tiles)
    if tiles.dtype != torch.uint8 or tiles.ndim != 5 or tiles.shape[-1] != 3:
        raise ValueError('tiles must be uint8 (bags, tiles, height, width, 3)')
    n, k = tiles.shape[:2]
    if n == 0 or k == 0:
        raise ValueError('tile bags cannot be empty')
    if lengths is None:
        lengths = torch.full((n,), k, device=device, dtype=torch.long)
    else:
        lengths = torch.as_tensor(lengths, device=device)
        if lengths.shape != (n,) or lengths.is_floating_point() or torch.any((lengths < 1) | (lengths > k)):
            raise ValueError('lengths must be integers between one and the padded bag size')
        lengths = lengths.long()
    images = tiles.reshape(n * k, *tiles.shape[2:]).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    return extractor.transform(images), _to_device(target, device), lengths, n, k


def _forward(head, features, lengths, batch, forward_fn):
    if forward_fn is not None:
        return forward_fn(head, features, lengths, batch)
    if getattr(head, 'use_lens', False):
        return head(features, lengths)
    if torch.any(lengths != features.shape[1]):
        raise ValueError('padded bags need a length-aware model or forward_fn')
    return head(features)


def _loss(loss_fn, output, targets):
    loss = loss_fn(output, targets)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError('loss_fn must return a finite scalar tensor')
    return loss


def predict_lora(extractor, head, batches, *, forward_fn=None, device=None):
    """Return detached model outputs and targets, preserving each batch's structure."""
    model = extractor.model
    device = torch.device(device or next(model.parameters()).device)
    model.to(device).eval()
    head.to(device).eval()
    outputs, targets = [], []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for batch in batches:
            images, target, lengths, n, k = _prepare(extractor, batch, device)
            features = encode_tiles(model, images).reshape(n, k, -1).float()
            outputs.append(_cpu(_forward(head, features, lengths, batch, forward_fn)))
            targets.append(_cpu(target))
    return {'outputs': outputs, 'targets': targets}


def build_lora_learner(extractor, head, train_batches, val_batches, *, config,
                       loss_fn=None, forward_fn=None, metrics=None, first_block=None,
                       rank=8, alpha=16, dropout=0.05, adapt=True, device=None,
                       seed=None, outdir=None, lr_adapter=5e-5, lr_head=None, categories=None):
    """Build a FastAI learner for raw-tile adapter training without fitting it."""
    from ._lora_fastai import build_learner
    return build_learner(
        extractor, head, train_batches, val_batches, config=config,
        loss_fn=loss_fn, forward_fn=forward_fn, metrics=metrics,
        first_block=first_block, rank=rank, alpha=alpha, dropout=dropout,
        adapt=adapt, device=device, seed=seed, outdir=outdir,
        lr_adapter=lr_adapter, lr_head=lr_head, categories=categories)


def train_lora(extractor, head, train_batches, *, loss_fn=None, forward_fn=None,
               val_batches=None, config=None, metrics=None, callbacks=None,
               epochs=None, first_block=None, rank=8, alpha=16, dropout=0.05,
               lr_adapter=5e-5, lr_head=None, weight_decay=1e-5, adapt=True,
               device=None, seed=None, outdir=None, outcomes='outcome',
               categories=None, prediction_fn=None, return_learner=False):
    """Fit adapters through Slideflow's FastAI trainer and save its standard results."""
    from ._lora_fastai import train
    return train(
        extractor, head, train_batches, loss_fn=loss_fn, forward_fn=forward_fn,
        val_batches=val_batches, config=config, metrics=metrics, callbacks=callbacks,
        epochs=epochs, first_block=first_block, rank=rank, alpha=alpha,
        dropout=dropout, lr_adapter=lr_adapter, lr_head=lr_head,
        weight_decay=weight_decay, adapt=adapt, device=device, seed=seed,
        outdir=outdir, outcomes=outcomes, categories=categories,
        prediction_fn=prediction_fn, return_learner=return_learner)
