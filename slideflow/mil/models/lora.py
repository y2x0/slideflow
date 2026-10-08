"""Tile encoder adapters with a selectable MIL aggregation head."""

import inspect

import torch
from torch import nn

from slideflow.model.lora import encode_tiles
from slideflow.model.extractors._lora import adapter_state_dict, init_lora


class LoRA(nn.Module):
    """Encode RGB tile bags and aggregate their features with a registered MIL head."""

    use_lens = True
    input_dtype = torch.uint8

    def __init__(self, n_in, n_out, *, encoder, head='nnmil', encoder_kwargs=None,
                 head_kwargs=None, first_block=None, rank=8, alpha=16,
                 dropout=0.05, adapt=True, tile_batch_size=32,
                 checkpoint_blocks=True):
        super().__init__()
        import slideflow as sf

        if n_in != 3:
            raise ValueError('LoRA requires RGB tile bags, not precomputed features')
        if not isinstance(tile_batch_size, int) or tile_batch_size < 1:
            raise ValueError('tile_batch_size must be a positive integer')
        if not isinstance(encoder, str) or not isinstance(head, str) or head == 'lora':
            raise ValueError('encoder and head must be registered names; head cannot be lora')
        options = dict(encoder_kwargs or {})
        options.setdefault('device', 'cpu')
        extractor = sf.build_feature_extractor(encoder, **options)
        self.encoder = extractor.model
        self.transform = extractor.transform
        self.num_features = extractor.num_features
        if not hasattr(self.encoder, 'blocks'):
            raise ValueError('LoRA requires a compatible timm vision transformer')
        self.first_block = max(0, len(self.encoder.blocks) - 8) if first_block is None else first_block
        if adapt:
            init_lora(self.encoder, first_block=self.first_block, rank=rank,
                      alpha=alpha, dropout=dropout)
        else:
            if hasattr(self.encoder, 'first_adapted'):
                raise ValueError('frozen control requires an encoder without adapters')
            self.encoder.requires_grad_(False)
        self.adapt = adapt
        self.tile_batch_size = tile_batch_size
        self.checkpoint_blocks = checkpoint_blocks
        head_class = sf.mil.get_model(head)
        if getattr(head_class, 'is_multimodal', False):
            raise ValueError('LoRA requires a single-input MIL head')
        self.head = head_class(self.num_features, n_out, **(head_kwargs or {}))
        self.uq_uses_softmax = 'uq_softmax' in inspect.signature(self.head.forward).parameters
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # keep the frozen backbone deterministic; only adapter dropout is trained
        self.encoder.eval()
        if self.adapt:
            for block in self.encoder.blocks[self.first_block:]:
                block.attn.qkv.drop.train(mode)
        return self

    def _features(self, bags, lens):
        if bags.ndim != 5 or bags.shape[-1] != 3:
            raise ValueError('LoRA expects RGB bags shaped (batch, tiles, height, width, 3)')
        n, k = bags.shape[:2]
        if n == 0 or k == 0:
            raise ValueError('tile bags cannot be empty')
        if lens is None:
            lens = torch.full((n,), k, dtype=torch.long, device=bags.device)
        if (lens.shape != (n,) or lens.is_floating_point()
                or torch.any((lens < 1) | (lens > k))):
            raise ValueError('bag lengths must be integers between one and the padded size')
        valid = torch.arange(k, device=bags.device)[None, :] < lens[:, None]
        images = bags[valid].permute(0, 3, 1, 2)
        chunks = [encode_tiles(self.encoder, self.transform(chunk.float()),
                               checkpoint_blocks=self.training and self.checkpoint_blocks)
                  for chunk in images.split(self.tile_batch_size)]
        encoded = torch.cat(chunks).float()
        features = encoded.new_zeros(n, k, self.num_features)
        features[valid] = encoded
        return features, lens

    def _head_forward(self, features, lens, **kwargs):
        if getattr(self.head, 'use_lens', False):
            return self.head(features, lens, **kwargs)
        if torch.any(lens != features.shape[1]):
            if kwargs:
                raise ValueError('attention and UQ for padded bags require a length-aware head')
            return torch.cat([self.head(bag[None, :length])
                              for bag, length in zip(features, lens)])
        return self.head(features, **kwargs)

    def forward(self, bags, lens=None, *, return_attention=False, uq=False,
                uq_softmax=None):
        if uq_softmax is None:
            uq_softmax = getattr(self, 'uq_apply_softmax', True)
        features, lens = self._features(bags, lens)
        params = inspect.signature(self.head.forward).parameters
        kwargs = {}
        if uq:
            if 'uq' not in params:
                raise ValueError('selected head does not support UQ')
            kwargs['uq'] = True
            if 'uq_softmax' in params:
                kwargs['uq_softmax'] = uq_softmax
        if return_attention and 'return_attention' in params:
            return self._head_forward(features, lens, return_attention=True, **kwargs)
        output = self._head_forward(features, lens, **kwargs)
        if return_attention:
            args = (features, lens) if getattr(self.head, 'use_lens', False) else (features,)
            return output, self.head.calculate_attention(*args)
        return output

    @property
    def calculate_attention(self):
        if not hasattr(self.head, 'calculate_attention'):
            raise AttributeError('selected head does not provide attention')
        return self._calculate_attention

    def _calculate_attention(self, bags, lens=None):
        return self(bags, lens, return_attention=True)[1]

    def export_adapters(self, path):
        """Save adapter tensors for use with the matching base feature extractor."""
        if not self.adapt:
            raise ValueError('frozen encoder has no adapters to export')
        torch.save(adapter_state_dict(self.encoder), path)
