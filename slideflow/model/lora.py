"""Train encoder adapters with a user-supplied prediction model and loss."""

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .extractors._lora import LoRAQKV, adapter_state_dict, apply_lora, init_lora

__all__ = ['LoRAQKV', 'init_lora', 'apply_lora', 'adapter_state_dict',
           'encode_tiles', 'train_lora', 'predict_lora']


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


def train_lora(extractor, head, train_batches, *, loss_fn, forward_fn=None,
               val_batches=None, epochs=8, first_block=None, rank=8, alpha=16,
               dropout=0.05, lr_adapter=5e-5, lr_head=2e-4, weight_decay=1e-5,
               adapt=True, device=None, seed=None, outdir=None):
    """Fit ViT adapters and any PyTorch bag model with the supplied scalar loss."""
    if epochs < 1 or not callable(loss_fn):
        raise ValueError('positive epochs and a callable loss_fn are required')
    if epochs > 1 and (iter(train_batches) is train_batches or
                       (val_batches is not None and iter(val_batches) is val_batches)):
        raise ValueError('multiple epochs require re-iterable training and validation loaders')
    root = Path(outdir) if outdir is not None else None
    if root is not None and root.exists():
        raise FileExistsError(f'output directory already exists: {root}')
    model = extractor.model
    device = torch.device(device or next(model.parameters()).device)
    first_block = max(0, len(model.blocks) - 8) if first_block is None else first_block
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    if adapt:
        init_lora(model, first_block=first_block, rank=rank, alpha=alpha, dropout=dropout)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        model.first_adapted = len(model.blocks)
    model.to(device)
    head.to(device)
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device=device, dtype=torch.float32)
    groups = []
    head_params = [param for param in head.parameters() if param.requires_grad]
    if head_params:
        groups.append({'params': head_params, 'lr': lr_head})
    if adapt:
        groups.append({'params': [param for param in model.parameters() if param.requires_grad], 'lr': lr_adapter})
    if not groups:
        raise ValueError('the model has no trainable parameters')
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        head.train()
        loss_sum, count = 0.0, 0
        for batch in train_batches:
            images, targets, lengths, n, k = _prepare(extractor, batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                features = encode_tiles(model, images, checkpoint_blocks=adapt).reshape(n, k, -1).float()
                output = _forward(head, features, lengths, batch, forward_fn)
                loss = _loss(loss_fn, output, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * n
            count += n
        if not count:
            raise ValueError('training batches are empty')
        schedule.step()
        row = {'epoch': epoch, 'train_loss': loss_sum / count, 'train_bags': count,
               'epoch_seconds': time.monotonic() - started}
        if val_batches is not None:
            model.eval()
            head.eval()
            val_sum, val_count = 0.0, 0
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                for batch in val_batches:
                    images, targets, lengths, n, k = _prepare(extractor, batch, device)
                    features = encode_tiles(model, images).reshape(n, k, -1).float()
                    output = _forward(head, features, lengths, batch, forward_fn)
                    val_sum += float(_loss(loss_fn, output, targets)) * n
                    val_count += n
            if not val_count:
                raise ValueError('validation batches are empty')
            row.update(val_loss=val_sum / val_count, val_bags=val_count)
        history.append(row)
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
        if adapt:
            torch.save(adapter_state_dict(model, first_block=first_block), root/'adapters.pt')
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, root/'head.pt')
        (root/'history.json').write_text(json.dumps({
            'encoder': getattr(extractor, 'tag', type(model).__name__),
            'head': type(head).__name__, 'loss': getattr(loss_fn, '__name__', type(loss_fn).__name__),
            'adapt': adapt, 'seed': seed, 'epochs': epochs,
            'first_block': first_block if adapt else None,
            'rank': rank if adapt else None, 'alpha': alpha if adapt else None,
            'dropout': dropout if adapt else None, 'history': history
        }, indent=2) + '\n')
    return history
