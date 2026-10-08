"""Train and reload LoRA models from existing RGB TFRecords."""

import copy
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

import slideflow as sf
from slideflow.mil.extensions import (
    build_fastai_learner, eval_mil, load_mil_config, mil_config, predict_mil,
)
from slideflow.mil._extension_data import TileBagDataset, resolve_tile_bags
from slideflow.mil.train import _fastai
from slideflow.test.lora_fastai_test import HEADS, config_for, dataset, extractor  # noqa: F401
from slideflow.tfrecord.writer import TFRecordWriter


def write_tiles(path, tiles, image_format='PNG', locations=True):
    writer = TFRecordWriter(str(path))
    try:
        for i, tile in enumerate(tiles):
            buffer = io.BytesIO()
            Image.fromarray(tile.numpy()).save(buffer, format=image_format)
            record = {'slide': (path.stem.encode(), 'byte'),
                      'image_raw': (buffer.getvalue(), 'byte')}
            if locations:
                record.update(loc_x=(i, 'int'), loc_y=(0, 'int'))
            writer.write(record)
    finally:
        writer.close()


def fixtures(tmp_path):
    folder = tmp_path / 'records'
    folder.mkdir()
    slides = [f'slide{i}' for i in range(10)]
    patients = {s: f'patient{i // 2}' for i, s in enumerate(slides)}
    labels = {s: (i // 2) % 2 for i, s in enumerate(slides)}
    for i, slide in enumerate(slides):
        tiles = torch.randint(0, 256, (i % 3 + 1, 16, 16, 3), dtype=torch.uint8)
        write_tiles(folder / f'{slide}.tfrecords', tiles)
    return folder, dataset(slides[:6], labels, patients), dataset(slides[6:], labels, patients)


def tile_config(head='nnmil', **kwargs):
    params = config_for(head, **kwargs).to_dict()
    return mil_config(**params)


@pytest.mark.parametrize('image_format', ['PNG', 'JPEG'])
@pytest.mark.parametrize('indexed', [False, True])
def test_decode_and_sample_without_writing_resources(tmp_path, image_format, indexed):
    path = tmp_path / 'slide.tfrecord'
    tiles = torch.full((7, 16, 16, 3), 123, dtype=torch.uint8)
    write_tiles(path, tiles, image_format, locations=False)
    if indexed:
        sf.util.tfrecord2idx.create_index(str(path))
    files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    bags = TileBagDataset([str(path)], bag_size=3, dtype=torch.uint8)
    with patch.object(sf.TFRecord, 'decode', autospec=True,
                      side_effect=sf.TFRecord.decode) as decode:
        actual, length = bags[0]
    assert decode.call_count == length == 3
    torch.testing.assert_close(actual, tiles[:3], atol=0, rtol=0)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == files
    clipped, length = TileBagDataset([str(path)], max_bag_size=2, dtype=torch.uint8)[0]
    assert clipped.shape == (2, 16, 16, 3) and length == 2


def test_grouped_records_and_padding(tmp_path):
    paths = []
    for i, count in enumerate([2, 3]):
        path = tmp_path / f'{i}.tfrecords'
        write_tiles(path, torch.full((count, 16, 16, 3), i + 1, dtype=torch.uint8))
        paths.append(str(path))
    tiles, length = TileBagDataset([paths], bag_size=7, dtype=torch.uint8)[0]
    assert length == 5 and tiles.shape == (7, 16, 16, 3)
    assert sorted(tiles[:, 0, 0, 0].tolist()) == [0, 0, 1, 1, 2, 2, 2]
    full, length = TileBagDataset([paths], dtype=torch.uint8)[0]
    assert length == 5 and full[:, 0, 0, 0].tolist() == [1, 1, 2, 2, 2]


@pytest.mark.parametrize('head', HEADS)
@pytest.mark.parametrize('level', ['slide', 'patient'])
def test_tfrecord_train_predict_reload(head, level, extractor, tmp_path):
    torch.manual_seed(17)
    folder, train, val = fixtures(tmp_path)
    config = tile_config(head, aggregation_level=level)
    if level == 'patient':
        config.batch_size = 3
    with patch.object(_fastai, 'build_learner', wraps=_fastai.build_learner) as build, \
            patch.object(_fastai, 'train', wraps=_fastai.train) as fit:
        learner = sf.mil.train_mil(config, train, val, 'label', folder,
                                   outdir=str(tmp_path / 'runs'), device='cpu')
    assert build.call_count == fit.call_count == 1
    train.get_bags.assert_not_called()
    val.get_bags.assert_not_called()
    assert len(learner.dls.train_ds) == (6 if level == 'slide' else 3)
    assert len(learner.dls.valid_ds) == (4 if level == 'slide' else 2)
    assert learner.dls.train.one_batch()[0].dtype == torch.uint8
    assert any(p.abs().sum() > 0 for name, p in learner.model.encoder.named_parameters()
               if '.Bq' in name)
    for name, p in learner.model.encoder.named_parameters():
        if not p.requires_grad:
            original = extractor.model.state_dict()[name.replace('.attn.qkv.base.', '.attn.qkv.')]
            torch.testing.assert_close(p, original, atol=0, rtol=0)
    history = pd.read_csv(learner.path / 'history.csv')
    assert np.isfinite(history[['train_loss', 'valid_loss', 'roc_auc_score']]).all().all()
    expected = pd.read_parquet(learner.path / 'predictions.parquet')
    restored, restored_config = sf.mil.load_model_weights(str(learner.path), strict=True)
    assert load_mil_config(str(learner.path), strict=True).to_dict() == config.to_dict()
    actual = predict_mil(restored, val, 'label', folder, config=restored_config)
    np.testing.assert_allclose(actual.filter(like='y_pred'), expected.filter(like='y_pred'), atol=1e-6)
    explicit = sf.mil.predict_mil(restored, val, 'label', resolve_tile_bags(folder),
                                  config=restored_config)
    np.testing.assert_allclose(explicit.filter(like='y_pred'), actual.filter(like='y_pred'), atol=0)
    saved = predict_mil(str(learner.path), val, 'label', folder)
    np.testing.assert_allclose(saved.filter(like='y_pred'), actual.filter(like='y_pred'), atol=0)
    assert len(list(folder.iterdir())) == 10


def test_tfrecord_evaluation_and_regression(extractor, tmp_path):
    folder, train, val = fixtures(tmp_path)
    train._filters = val._filters = {}
    config = tile_config(loss='mse', fit_one_cycle=False)
    config.model_config.model_kwargs['adapt'] = False
    learner = sf.mil.train_mil(config, train, val, 'score', folder,
                               outdir=str(tmp_path / 'runs'), device='cpu')
    frame = eval_mil(str(learner.path), val, 'score', folder, outdir=str(tmp_path / 'eval'))
    expected = pd.read_parquet(learner.path / 'predictions.parquet')
    np.testing.assert_allclose(frame.filter(like='y_pred'), expected.filter(like='y_pred'), atol=1e-6)


def test_second_train_mil_freezes_saved_adapters(extractor, tmp_path):
    folder, train, val = fixtures(tmp_path)
    config = tile_config()
    first = sf.mil.train_mil(config, train, val, 'label', folder,
                             outdir=str(tmp_path / 'joint'), device='cpu')
    adapter_path = tmp_path / 'adapters.pt'
    head_path = tmp_path / 'head.pt'
    first.model.export_adapters(adapter_path)
    torch.save(first.model.head.state_dict(), head_path)
    frozen = tile_config()
    frozen.model_config.model_kwargs.update(adapt=False, adapters=str(adapter_path),
                                            head_weights=str(head_path))
    before = frozen.build_model(3, 2)
    for name, p in before.state_dict().items():
        torch.testing.assert_close(p, first.model.state_dict()[name], atol=0, rtol=0)
    encoder_before = copy.deepcopy(before.encoder.state_dict())
    head_before = copy.deepcopy(before.head.state_dict())
    second = sf.mil.train_mil(frozen, train, val, 'label', folder,
                              outdir=str(tmp_path / 'head_only'), device='cpu')
    assert all(not p.requires_grad and p.grad is None for p in second.model.encoder.parameters())
    for name, value in second.model.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[name], atol=0, rtol=0)
    assert any(not torch.equal(value, head_before[name])
               for name, value in second.model.head.state_dict().items())
    second.model.export_adapters(tmp_path / 'frozen_adapters.pt')
    params = json.loads((second.path / 'mil_params.json').read_text())['params']['model_kwargs']
    assert params['adapt'] is False and params['use_adapters'] is True
    assert 'adapters' not in params and 'head_weights' not in params
    adapter_path.unlink()
    head_path.unlink()
    restored, restored_config = sf.mil.load_model_weights(str(second.path), strict=True)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], atol=0, rtol=0)
    actual = predict_mil(restored, val, 'label', folder, config=restored_config)
    expected = pd.read_parquet(second.path / 'predictions.parquet')
    np.testing.assert_allclose(actual.filter(like='y_pred'), expected.filter(like='y_pred'), atol=1e-6)


def test_worker_and_explicit_paths(extractor, tmp_path):
    folder, train, val = fixtures(tmp_path)
    config = tile_config()
    config.model_config.num_workers = 1
    learner = build_fastai_learner(config, train, val, 'label', [folder],
                                   outdir=str(tmp_path), device='cpu')
    tiles, lengths, targets = learner.dls.train.one_batch()
    assert tiles.dtype == torch.uint8 and len(tiles) == len(lengths) == len(targets) == 2


def test_directory_errors_and_existing_pt_support(tmp_path):
    with pytest.raises(ValueError, match='No .tfrecord'):
        resolve_tile_bags(tmp_path)
    path = tmp_path / 'slide.pt'
    tiles = torch.ones(2, 16, 16, 3, dtype=torch.uint8)
    torch.save(tiles, path)
    assert resolve_tile_bags([tmp_path]) == [str(path)]
    actual, length = TileBagDataset([str(path)], dtype=torch.uint8)[0]
    torch.testing.assert_close(actual, tiles)
    assert length == 2
    write_tiles(tmp_path / 'slide.tfrecords', tiles)
    with pytest.raises(ValueError, match='both TFRecords and .pt'):
        resolve_tile_bags(tmp_path)
    with pytest.raises(ValueError, match='cannot mix'):
        TileBagDataset([[str(path), str(tmp_path / 'slide.tfrecords')]], dtype=torch.uint8)[0]


def test_empty_records(tmp_path):
    path = tmp_path / 'empty.tfrecords'
    path.touch()
    with pytest.raises(ValueError, match='cannot be empty'):
        TileBagDataset([str(path)], dtype=torch.uint8)[0]
