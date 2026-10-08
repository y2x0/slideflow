"""Check adapter insertion and checkpoint compatibility."""

import copy
import tempfile
from pathlib import Path

import pytest
import timm
import torch

from slideflow.mil.models import NNMIL
from slideflow.model.lora import (
    adapter_state_dict, apply_lora, encode_tiles, init_lora,
)


class Extractor:
    tag = 'tiny_vit'
    transform = staticmethod(lambda images: images / 255)

    def __init__(self):
        self.model = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0,
            img_size=16, patch_size=8, embed_dim=16, depth=2, num_heads=2)


def test_zero_init_and_adapter_roundtrip():
    torch.manual_seed(3)
    model = Extractor().model.eval()
    images = torch.randn(2, 3, 16, 16)
    reference = encode_tiles(model, images)
    base = copy.deepcopy(model)
    init_lora(model, first_block=1, rank=2, alpha=4)
    model.eval()
    torch.testing.assert_close(encode_tiles(model, images), reference)
    assert all(param.requires_grad == ('attn.qkv.A' in name or 'attn.qkv.B' in name)
               for name, param in model.named_parameters())
    with pytest.raises(ValueError):
        init_lora(model, first_block=1, rank=2)
    apply_lora(base, adapter_state_dict(model), first_block=1, rank=2, alpha=4)
    base.eval()
    torch.testing.assert_close(encode_tiles(base, images), reference)


@pytest.mark.parametrize('legacy', [False, True])
def test_nnmil_checkpoint_loader(legacy):
    model = NNMIL(16, 2, hidden_dim=8, stride_divisor=2).eval()
    state = model.state_dict()
    if legacy:
        rename = {'attention_V': 'V', 'attention_U': 'U', 'attention_w': 'w',
                  'head': 'cls', 'eval_subsets': 'chunks'}
        state = {rename.get(key.split('.')[0], key.split('.')[0])
                 + key[len(key.split('.')[0]):]: value for key, value in state.items()}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'head.pt'
        torch.save(state, path)
        loaded = NNMIL.from_checkpoint(path)
    features = torch.randn(2, 3, 16)
    torch.testing.assert_close(loaded(features), model(features), atol=0, rtol=0)
