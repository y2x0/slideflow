"""Check LoRA training uses the same FastAI callbacks, metrics and result files."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from fastai.callback.core import Callback
from fastai.learner import Learner

import slideflow as sf
from slideflow.mil import mil_config
from slideflow.mil.train import _fastai
from slideflow.model.lora import build_lora_learner, train_lora, predict_lora
from slideflow.test.lora_train_test import Extractor, batches


class Events(Callback):
    def before_fit(self):
        self.started = True

    def after_backward(self):
        self.adapter_gradient = any(param.grad is not None and param.grad.abs().sum() > 0
                                    for name, param in self.model.named_parameters() if '.Bq' in name)


@pytest.mark.parametrize('architecture', ['nnmil', 'attention_mil', 'bistro.transformer'])
def test_native_metrics_callbacks_and_results(architecture, tmp_path):
    torch.manual_seed(19)
    kwargs = {'hidden_dim': 8} if architecture == 'nnmil' else (
        {'z_dim': 8, 'dropout_p': 0} if architecture == 'attention_mil' else
        {'dim': 16, 'depth': 1, 'heads': 2, 'dim_head': 8, 'mlp_dim': 16})
    config = mil_config(architecture, lr=1e-3, epochs=2, model_kwargs=kwargs)
    extractor = Extractor()
    base = copy.deepcopy(extractor.model.state_dict())
    head = config.build_model(16, 2)
    train_loader = batches()
    validation = []
    for index, batch in enumerate(batches()):
        validation.append({'tiles': batch[0], 'targets': batch[1],
                           'slide': [f'slide_{index * 2}', f'slide_{index * 2 + 1}'],
                           'patient': [f'patient_{index * 2}', f'patient_{index * 2 + 1}']})
    events = Events()
    root = tmp_path / architecture
    with patch.object(_fastai, 'train', wraps=_fastai.train) as shared_train:
        learner = train_lora(extractor, head, train_loader, val_batches=validation,
                             config=config, first_block=1, rank=2, alpha=4,
                             outdir=root, outcomes='label', categories=['a', 'b'],
                             callbacks=[events], return_learner=True)
    assert isinstance(learner, Learner)
    assert shared_train.call_count == 1
    assert events.started and events.adapter_gradient
    assert 'roc_auc_score' in learner.recorder.metric_names
    history = pd.read_csv(root / 'history.csv')
    assert len(history) == config.epochs
    assert np.isfinite(history[['train_loss', 'valid_loss', 'roc_auc_score']]).all().all()
    assert (root / 'models' / 'best_valid.pth').exists()
    assert len(list(root.glob('*.png'))) + len(list(root.glob('*.svg'))) > 0
    frame = pd.read_parquet(root / 'predictions.parquet')
    assert frame.slide.tolist() == [f'slide_{i}' for i in range(4)]
    assert frame.patient.tolist() == [f'patient_{i}' for i in range(4)]
    assert {'label-y_true', 'label-y_pred0', 'label-y_pred1'}.issubset(frame.columns)
    np.testing.assert_allclose(frame[['label-y_pred0', 'label-y_pred1']].sum(axis=1), 1)
    reference = predict_lora(extractor, head, validation)
    before = torch.cat(reference['outputs'])
    with torch.no_grad():
        for param in learner.model.parameters():
            if param.requires_grad:
                param.add_(1)
    learner.load('best_valid', with_opt=False)
    restored = torch.cat(predict_lora(extractor, head, validation)['outputs'])
    torch.testing.assert_close(restored, before, atol=0, rtol=0)
    for name, param in extractor.model.named_parameters():
        if not param.requires_grad:
            base_name = name.replace('.attn.qkv.base.', '.attn.qkv.')
            torch.testing.assert_close(param, base[base_name], atol=0, rtol=0)
    params = json.loads((root / 'mil_params.json').read_text())
    assert params['training_input'] == 'raw_tiles'
    assert params['input_shape'] == 16
    assert params['output_shape'] == 2
    loaded_head, loaded_config = sf.mil.load_model_weights(str(root), strict=True)
    features = torch.randn(2, 3, 16)
    head.eval()
    loaded_head.cpu().eval()
    lengths = torch.full((2,), 3)
    expected = head(features, lengths) if getattr(head, 'use_lens', False) else head(features)
    actual = loaded_head(features, lengths) if getattr(loaded_head, 'use_lens', False) else loaded_head(features)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert loaded_config.model_config.model == architecture


def test_regression_metrics_and_config_preservation(tmp_path):
    config = mil_config('nnmil', loss='mse', epochs=1, lr=1e-3, fit_one_cycle=False,
                        model_kwargs={'hidden_dim': 8})
    targets = torch.tensor([[0.], [0.5], [1.], [1.5]])
    loader = batches(targets)
    learner = train_lora(Extractor(), config.build_model(16, 1), loader,
                         config=config, val_batches=loader, outdir=tmp_path / 'regression',
                         outcomes='measurement', return_learner=True)
    assert 'mse' in learner.recorder.metric_names
    assert 'pearsonr' in learner.recorder.metric_names
    assert {'measurement-y_true', 'measurement-y_pred'}.issubset(learner.lora_predictions.columns)
    assert config.save_monitor == 'valid_loss'
    assert config.epochs == 1


def test_build_without_fitting_and_custom_metrics(tmp_path):
    from fastai.vision.all import accuracy
    config = mil_config('nnmil', epochs=1, lr=1e-3, model_kwargs={'hidden_dim': 8})
    extractor, loader = Extractor(), batches()
    learner = build_lora_learner(extractor, config.build_model(16, 2), loader, loader,
                                 config=config, metrics=[accuracy], outdir=tmp_path / 'custom',
                                 lr_adapter=1e-5, lr_head=1e-3)
    assert isinstance(learner, Learner)
    assert not (learner.path / 'history.csv').exists()
    assert learner.lora_lrs == [1e-5, 1e-3]
    _fastai.train(learner, learner.lora_config, lr=learner.lora_lrs)
    assert 'accuracy' in pd.read_csv(learner.path / 'history.csv').columns


def test_structured_targets_use_fastai_with_custom_loss(tmp_path):
    config = mil_config('nnmil', epochs=1, lr=1e-3, model_kwargs={'hidden_dim': 8})
    extractor = Extractor()
    head = config.build_model(16, 2)
    tiles, targets = next(iter(batches()))
    loader = [{'tiles': tiles, 'targets': {'class': targets}}]

    def loss_fn(outputs, targets):
        return nn.functional.cross_entropy(outputs, targets['class'])

    def prediction_fn(outputs, targets, ids):
        scores = outputs.softmax(-1).cpu().numpy()
        return pd.DataFrame({'slide': ids, 'label-y_true': targets['class'].cpu().numpy(),
                             'label-y_pred0': scores[:, 0], 'label-y_pred1': scores[:, 1]})

    learner = train_lora(extractor, head, loader, config=config, loss_fn=loss_fn,
                         metrics=[], val_batches=loader, outcomes='label',
                         prediction_fn=prediction_fn, outdir=tmp_path / 'structured',
                         return_learner=True)
    assert (learner.path / 'predictions.parquet').exists()
    assert isinstance(learner.lora_validation['targets'], dict)


def test_weighted_loss_with_class_absent_from_training(tmp_path):
    config = mil_config('nnmil', epochs=1, lr=1e-3, model_kwargs={'hidden_dim': 8})
    head = config.build_model(16, 3)
    loader = batches()
    val = batches(torch.tensor([0, 1, 2, 2]))
    learner = build_lora_learner(Extractor(), head, loader, val, config=config,
                                 categories=['a', 'b', 'c'], outdir=tmp_path / 'missing_class')
    _fastai.train(learner, learner.lora_config, lr=learner.lora_lrs)
    weights = learner.loss_func.fn.weight
    torch.testing.assert_close(weights, torch.tensor([0.4, 0.4, 0.2]))
    assert np.isfinite(pd.read_csv(learner.path / 'history.csv')['roc_auc_score']).all()


def test_build_reads_label_metadata_without_decoding_tiles(tmp_path):
    class Dataset(torch.utils.data.Dataset):
        targets = torch.tensor([0, 1, 0, 1])

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            raise RuntimeError('tiles should only be loaded when fitting')

    loader = torch.utils.data.DataLoader(Dataset(), batch_size=2)
    config = mil_config('nnmil', epochs=1, lr=1e-3, model_kwargs={'hidden_dim': 8})
    learner = build_lora_learner(Extractor(), config.build_model(16, 2), loader, loader,
                                 config=config, outdir=tmp_path / 'lazy')
    assert isinstance(learner, Learner)
    torch.testing.assert_close(learner.loss_func.fn.weight, torch.tensor([0.5, 0.5]))
