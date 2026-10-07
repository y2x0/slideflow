"""Check generic raw-tile adapter training with small timm transformers."""

import copy
import tempfile
from pathlib import Path

import pytest
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from slideflow.mil.models import NNMIL, Attention_MIL
from slideflow.mil.models.bistro import Transformer
from slideflow.model.lora import (
    LoRAQKV, adapter_state_dict, apply_lora, encode_tiles, init_lora,
    predict_lora, train_lora,
)


class Extractor:
    tag = 'tiny_vit'
    transform = staticmethod(lambda images: images / 255)

    def __init__(self):
        self.model = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0,
            img_size=16, patch_size=8, embed_dim=16, depth=2, num_heads=2)


def batches(targets=None):
    tiles = torch.randint(0, 255, (4, 3, 16, 16, 3), dtype=torch.uint8)
    labels = torch.tensor([0, 1, 0, 1]) if targets is None else targets
    return DataLoader(TensorDataset(tiles, labels), batch_size=2)


class MeanPool(nn.Module):
    def __init__(self, outputs=2):
        super().__init__()
        self.projection = nn.Linear(16, outputs)

    def forward(self, features):
        return self.projection(features.mean(1))


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


@pytest.mark.parametrize('architecture', ['nnmil', 'attention_mil', 'bistro', 'custom'])
def test_classification_with_different_heads(architecture, tmp_path):
    torch.manual_seed(4)
    head = {
        'nnmil': lambda: NNMIL(16, 2, hidden_dim=8, dropout_p=0),
        'attention_mil': lambda: Attention_MIL(16, 2, z_dim=8, dropout_p=0),
        'bistro': lambda: Transformer(16, 2, dim=16, depth=1, heads=2,
                                      dim_head=8, mlp_dim=16),
        'custom': MeanPool,
    }[architecture]()
    original = copy.deepcopy(head.state_dict())
    extractor = Extractor()
    base = copy.deepcopy(extractor)
    loader = batches()
    output = tmp_path / architecture
    history = train_lora(
        extractor, head, loader, loss_fn=nn.CrossEntropyLoss(),
        val_batches=loader, epochs=2, first_block=1, rank=2, alpha=4,
        device='cpu', seed=123, outdir=output)
    assert len(history) == 2
    assert all(row['train_bags'] == row['val_bags'] == 4 for row in history)
    assert all(torch.isfinite(torch.tensor(row['val_loss'])) for row in history)
    assert any(not torch.equal(value, original[key]) for key, value in head.state_dict().items())
    state = torch.load(output / 'adapters.pt', weights_only=True)
    assert any(value.abs().sum() > 0 for key, value in state.items() if '.B' in key)
    saved_head = copy.deepcopy(head)
    saved_head.load_state_dict(torch.load(output / 'head.pt', weights_only=True))
    apply_lora(base.model, state, first_block=1, rank=2, alpha=4)
    assert isinstance(base.model.blocks[1].attn.qkv, LoRAQKV)
    prediction = predict_lora(extractor, saved_head, loader)
    assert set(prediction) == {'outputs', 'targets'}
    assert torch.cat(prediction['outputs']).shape == (4, 2)
    restored = predict_lora(base, saved_head, loader)
    torch.testing.assert_close(torch.cat(prediction['outputs']),
                               torch.cat(restored['outputs']), atol=0, rtol=0)
    assert torch.equal(torch.cat(prediction['targets']), torch.tensor([0, 1, 0, 1]))


@pytest.mark.parametrize('loss_fn', [nn.MSELoss(), nn.BCEWithLogitsLoss()])
def test_other_objectives_and_frozen_control(loss_fn, tmp_path):
    extractor = Extractor()
    before = copy.deepcopy(extractor.model.state_dict())
    head = MeanPool(1)
    before_head = copy.deepcopy(head.state_dict())
    history = train_lora(extractor, head, batches(torch.tensor([[0.], [1.], [0.], [1.]])),
                         loss_fn=loss_fn, epochs=1, adapt=False, device='cpu',
                         outdir=tmp_path / 'control')
    assert torch.isfinite(torch.tensor(history[0]['train_loss']))
    assert not (tmp_path / 'control' / 'adapters.pt').exists()
    for key, value in extractor.model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert any(not torch.equal(value, before_head[key]) for key, value in head.state_dict().items())


def test_structured_outputs_targets_and_callback():
    class MultiTask(MeanPool):
        def forward(self, features, lengths):
            mask = torch.arange(features.shape[1])[None, :] < lengths[:, None]
            pooled = (features * mask[..., None]).sum(1) / lengths[:, None]
            output = self.projection(pooled)
            return {'classes': output, 'continuous': output[:, :1]}, lengths

    extractor = Extractor()
    head = MultiTask()
    tiles = next(iter(batches()))[0]
    loader = [{'tiles': tiles, 'lengths': torch.tensor([1, 2]),
               'targets': {'classes': torch.tensor([0, 1]),
                           'continuous': torch.tensor([[0.], [1.]], dtype=torch.float64)}}]

    def forward_fn(model, features, lengths, batch):
        assert batch['tiles'] is tiles
        return model(features, lengths)

    def loss_fn(output, targets):
        assert targets['continuous'].dtype == torch.float32
        return (nn.functional.cross_entropy(output[0]['classes'], targets['classes'])
                + nn.functional.mse_loss(output[0]['continuous'], targets['continuous']))

    train_lora(extractor, head, loader, loss_fn=loss_fn, forward_fn=forward_fn,
               val_batches=loader, epochs=1, device='cpu')
    result = predict_lora(extractor, head, loader, forward_fn=forward_fn)
    assert isinstance(result['outputs'][0], tuple)
    assert result['outputs'][0][0]['classes'].shape == (2, 2)
    assert not result['outputs'][0][0]['classes'].requires_grad
    torch.testing.assert_close(result['outputs'][0][1], torch.tensor([1, 2]))


def test_reproducibility_and_float64_loss_weights():
    torch.manual_seed(9)
    base, head, loader = Extractor(), MeanPool(), batches()
    states = []
    for _ in range(2):
        extractor = copy.deepcopy(base)
        train_lora(extractor, copy.deepcopy(head), loader,
                   loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1., 2.], dtype=torch.float64)),
                   epochs=1, seed=123, device='cpu')
        states.append(adapter_state_dict(extractor.model))
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


def test_invalid_batches_and_output_protection(tmp_path):
    extractor, head, loader = Extractor(), MeanPool(), batches()
    with pytest.raises(FileExistsError):
        train_lora(extractor, head, loader, loss_fn=nn.CrossEntropyLoss(), outdir=tmp_path)
    with pytest.raises(ValueError, match='re-iterable'):
        train_lora(extractor, head, iter(loader), loss_fn=nn.CrossEntropyLoss(), epochs=2)
    with pytest.raises(ValueError, match='length-aware'):
        batch = next(iter(loader))
        predict_lora(extractor, head, [(batch[0], batch[1], torch.tensor([1, 2]))])
    with pytest.raises(ValueError, match='finite scalar'):
        train_lora(extractor, head, loader, loss_fn=lambda output, target: output,
                   epochs=1, adapt=False)


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
