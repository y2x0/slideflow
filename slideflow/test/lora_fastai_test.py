"""Exercise raw-tile LoRA through the ordinary MIL training pipeline."""

import slideflow.mil.extensions

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

import slideflow as sf
from slideflow.mil import mil_config
from slideflow.mil._extension_data import InstanceBagDataset as BagDataset
from slideflow.mil.train import _fastai
from slideflow.test.lora_train_test import Extractor


@pytest.fixture
def extractor():
    base = Extractor()
    base.num_features = 16
    with patch('slideflow.build_feature_extractor', side_effect=lambda *a, **kw: copy.deepcopy(base)):
        yield base


def dataset(slides, labels, patients):
    result = Mock(spec=sf.Dataset)
    result.slides.return_value = slides
    result.patients.return_value = {s: patients[s] for s in slides}

    def get_labels(outcomes, format='name', use_float=False):
        values = {s: ([float(labels[s])] if use_float else
                      labels[s] if format == 'id' else str(labels[s])) for s in slides}
        return values, sorted(set(values.values())) if not use_float else []

    result.labels.side_effect = get_labels
    result.get_bags.side_effect = lambda folder: np.array([str(Path(folder) / f'{s}.pt') for s in slides])
    return result


def fixtures(tmp_path):
    folder = tmp_path / 'tiles'
    folder.mkdir()
    slides = [f'slide{i}' for i in range(10)]
    patients = {s: f'patient{i // 2}' for i, s in enumerate(slides)}
    labels = {s: (i // 2) % 2 for i, s in enumerate(slides)}
    for i, slide in enumerate(slides):
        torch.save(torch.randint(0, 256, (i % 3 + 1, 16, 16, 3), dtype=torch.uint8),
                   folder / f'{slide}.pt')
    return (folder, dataset(slides[:6], labels, patients),
            dataset(slides[6:], labels, patients))


def config_for(head='nnmil', **kwargs):
    options = {'hidden_dim': 8, 'dropout_p': 0} if head == 'nnmil' else (
        {'z_dim': 8, 'dropout_p': 0} if head == 'attention_mil' else
        {'dim': 16, 'depth': 1, 'heads': 2, 'dim_head': 8, 'mlp_dim': 16})
    return mil_config('lora', epochs=2, lr=1e-3, batch_size=2,
                      bag_size=4, drop_last=False, num_workers=0,
                      model_kwargs=dict(encoder='tiny', head=head, head_kwargs=options,
                                        first_block=1, rank=2, alpha=4, dropout=0,
                                        tile_batch_size=2), **kwargs)


@pytest.mark.parametrize('head', ['nnmil', 'attention_mil', 'bistro.transformer'])
@pytest.mark.parametrize('level', ['slide', 'patient'])
def test_train_mil_roundtrip(head, level, extractor, tmp_path):
    torch.manual_seed(17)
    folder, train, val = fixtures(tmp_path)
    config = config_for(head, aggregation_level=level)
    # AMIL's batch norm needs at least two training bags.
    if level == 'patient':
        config.batch_size = 3
    with patch.object(_fastai, 'build_learner', wraps=_fastai.build_learner) as build, \
            patch.object(_fastai, 'train', wraps=_fastai.train) as fit, \
            patch.object(config, 'build_model', wraps=config.build_model) as model_build:
        learner = sf.mil.train_mil(config, train, val, 'label', str(folder),
                                   outdir=str(tmp_path / 'runs'), device='cpu')
    assert build.call_count == fit.call_count == model_build.call_count == 1
    assert len(learner.dls.train_ds) == (6 if level == 'slide' else 3)
    assert len(learner.dls.valid_ds) == (4 if level == 'slide' else 2)
    assert learner.dls.train.one_batch()[0].dtype == torch.uint8
    model = learner.model.eval()
    base = extractor.model.state_dict()
    for name, parameter in model.encoder.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, base[name.replace('.attn.qkv.base.', '.attn.qkv.')],
                                       rtol=0, atol=0)
    assert any(p.abs().sum() > 0 for name, p in model.encoder.named_parameters() if '.Bq' in name)
    root = learner.path
    history = pd.read_csv(root / 'history.csv')
    assert len(history) == 2
    assert np.isfinite(history[['train_loss', 'valid_loss', 'roc_auc_score']]).all().all()
    frame = pd.read_parquet(root / 'predictions.parquet')
    assert len(frame) == (4 if level == 'slide' else 2)
    assert set(frame[level]) == set(val.slides() if level == 'slide' else val.patients().values())
    params = json.loads((root / 'mil_params.json').read_text())
    assert params['params']['model'] == 'lora'
    assert params['params']['model_kwargs']['head'] == head
    restored, restored_config = sf.mil.load_model_weights(str(root), strict=True)
    actual = sf.mil.predict_mil(restored, val, 'label', str(folder), config=restored_config)
    np.testing.assert_allclose(actual.filter(like='y_pred'), frame.filter(like='y_pred'), atol=1e-6)
    model.export_adapters(root / 'adapters.pt')
    assert torch.load(root / 'adapters.pt', weights_only=True)


@pytest.mark.parametrize('head', ['nnmil', 'attention_mil', 'bistro.transformer'])
def test_padding_does_not_change_predictions(head, extractor):
    model = config_for(head).build_model(3, 2).eval()
    tiles = torch.randint(0, 256, (1, 2, 16, 16, 3), dtype=torch.uint8)
    padded = torch.cat([tiles, torch.full_like(tiles, 255)], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(model(tiles), model(padded, torch.tensor([2])), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('head', ['nnmil', 'attention_mil', 'bistro.transformer'])
@pytest.mark.parametrize('checkpoint_blocks', [False, True])
def test_matched_control_and_joint_gradients(head, checkpoint_blocks, extractor):
    torch.manual_seed(23)
    config = config_for(head)
    config.model_config.model_kwargs['checkpoint_blocks'] = checkpoint_blocks
    adapted = config.build_model(3, 2)
    config.model_config.model_kwargs['adapt'] = False
    frozen = config.build_model(3, 2)
    # Adapter initialization consumes RNG draws before the head is constructed.
    frozen.head.load_state_dict(adapted.head.state_dict())
    tiles = torch.randint(0, 256, (2, 3, 16, 16, 3), dtype=torch.uint8)
    lens = torch.tensor([3, 3])
    adapted.eval()
    frozen.eval()
    with torch.no_grad():
        torch.testing.assert_close(adapted(tiles, lens), frozen(tiles, lens),
                                   atol=1e-6, rtol=1e-5)

    for model in (adapted, frozen):
        model.train()
        loss = torch.nn.functional.cross_entropy(model(tiles, lens), torch.tensor([0, 1]))
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.head.parameters())
        assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    adapter_grads = [p.grad for p in adapted.encoder.parameters() if p.requires_grad]
    assert adapter_grads and all(g is not None and torch.isfinite(g).all() for g in adapter_grads)
    assert any(g.abs().sum() > 0 for g in adapter_grads)
    assert all(not p.requires_grad and p.grad is None for p in frozen.encoder.parameters())


def test_patient_bag_loading_preserves_all_slides(tmp_path):
    paths = []
    for i, count in enumerate([2, 3]):
        path = tmp_path / f'{i}.pt'
        torch.save(torch.full((count, 2, 2, 3), i + 1, dtype=torch.uint8), path)
        paths.append(str(path))
    bags = BagDataset(np.array([paths], dtype=object), bag_size=7, dtype=torch.uint8)
    tiles, length = bags[0]
    assert length == 5 and tiles.shape == (7, 2, 2, 3)
    assert sorted(tiles[:, 0, 0, 0].tolist()) == [0, 0, 1, 1, 2, 2, 2]


@pytest.mark.parametrize('level', ['slide'])
def test_regression_and_frozen_control(level, extractor, tmp_path):
    folder, train, val = fixtures(tmp_path)
    config = config_for(loss='mse', fit_one_cycle=False, aggregation_level=level)
    config.model_config.model_kwargs['adapt'] = False
    learner = sf.mil.train_mil(config, train, val, 'score', str(folder),
                               outdir=str(tmp_path / 'regression'), device='cpu')
    for name, value in learner.model.encoder.state_dict().items():
        torch.testing.assert_close(value, extractor.model.state_dict()[name], atol=0, rtol=0)
    history = pd.read_csv(learner.path / 'history.csv')
    assert np.isfinite(history[['train_loss', 'valid_loss', 'mse']]).all().all()
    with pytest.raises(ValueError, match='no adapters'):
        learner.model.export_adapters(tmp_path / 'invalid.pt')


def test_invalid_feature_input(extractor):
    with pytest.raises(ValueError, match='RGB'):
        config_for().build_model(16, 2)
    model = config_for().build_model(3, 2)
    with pytest.raises(ValueError, match='RGB'):
        model(torch.randn(2, 4, 3))


def test_patient_regression_reports_native_limit():
    with pytest.raises(ValueError, match='unchanged Slideflow'):
        config_for(loss='mse', aggregation_level='patient')


@pytest.mark.parametrize('head', ['nnmil', 'attention_mil'])
@pytest.mark.parametrize('loss', ['cross_entropy', 'mse'])
def test_uq_outputs(head, loss, extractor):
    config = config_for(head, loss=loss)
    model = config.build_model(3, 2 if loss == 'cross_entropy' else 1).eval()
    tiles = torch.randint(0, 256, (1, 3, 16, 16, 3), dtype=torch.uint8)
    torch.manual_seed(9)
    expected, std = model(tiles, uq=True, uq_softmax=loss == 'cross_entropy')
    torch.manual_seed(9)
    actual, attention, actual_std = config.batched_predict(model, tiles, uq=True, device='cpu')
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_std, std)
