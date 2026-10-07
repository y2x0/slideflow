"""Build raw-tile learners with Slideflow's shared FastAI training machinery."""

import copy
from collections.abc import Iterator
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from fastai.callback.core import Callback
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.learner import Metric

from slideflow.mil import mil_config
from slideflow.mil.train import _fastai, _log_mil_params
from .lora import _cpu, _forward, _loss, _prepare, _to_device, encode_tiles
from .extractors._lora import adapter_state_dict, init_lora


class _Batches:
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        for batch in self.source:
            targets = batch['targets'] if isinstance(batch, dict) else batch[1]
            yield batch, _to_device(targets, 'cpu')


class _Loss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.fn = loss_fn

    def forward(self, outputs, targets):
        return _loss(self.fn, outputs, targets)


class LoRAModel(nn.Module):
    """Encode raw tile bags and pass their features to the caller's model."""
    def __init__(self, extractor, head, forward_fn=None, checkpoint_blocks=True):
        super().__init__()
        self.encoder = extractor.model
        self.head = head
        self.transform = extractor.transform
        self.forward_fn = forward_fn
        self.checkpoint_blocks = checkpoint_blocks

    @property
    def model(self):
        return self.encoder

    def forward(self, batch):
        device = next(self.encoder.parameters()).device
        images, _, lengths, n, k = _prepare(self, batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            features = encode_tiles(self.encoder, images,
                                    checkpoint_blocks=self.training and self.checkpoint_blocks)
            output = _forward(self.head, features.reshape(n, k, -1).float(),
                              lengths, batch, self.forward_fn)
        return _float_output(output)


def _float_output(value):
    if isinstance(value, dict):
        return {key: _float_output(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_float_output(item) for item in value)
    return value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value


def _splitter(model):
    groups = [[param for param in part.parameters() if param.requires_grad]
              for part in (model.encoder, model.head)]
    return [group for group in groups if group]


class _CategoricalMetric(Metric):
    """Give Slideflow's ROC metric the one-hot targets used by saved-feature loaders."""
    def __init__(self, metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    @property
    def name(self):
        return self.metric.name

    @property
    def value(self):
        return self.metric.value

    def accumulate(self, learn):
        pred, targets = learn.pred, learn.y
        old_pred, old_yb = learn.pred, learn.yb
        if pred.ndim == 1 or pred.shape[-1] == 1:
            pred = pred.reshape(-1, 1)
            learn.pred = torch.cat([torch.zeros_like(pred), pred], dim=-1)
        else:
            learn.pred = pred
        if targets.ndim == 1 or targets.shape[-1] == 1:
            targets = nn.functional.one_hot(targets.reshape(-1).long(), learn.pred.shape[-1]).float()
        learn.yb = (targets,)
        try:
            self.metric.accumulate(learn)
        finally:
            learn.pred, learn.yb = old_pred, old_yb


class _History(Callback):
    order = 65

    def before_fit(self):
        self.learn.lora_history = []

    def before_epoch(self):
        self.started = time.monotonic()
        self.counts = {'train': 0, 'val': 0}

    def after_batch(self):
        batch = self.xb[0]
        tiles = batch['tiles'] if isinstance(batch, dict) else batch[0]
        self.counts['train' if self.training else 'val'] += len(tiles)
        if isinstance(self.pred, torch.Tensor):
            self.learn.lora_output_shape = self.pred.shape[-1] if self.pred.ndim > 1 else 1

    def after_epoch(self):
        names = self.recorder.metric_names[1:-1]
        values = self.recorder.values[-1]
        row = {name: float(value) if value is not None else None
               for name, value in zip(names, values)}
        row['val_loss'] = row.pop('valid_loss', None)
        row.update(epoch=self.epoch + 1, train_bags=self.counts['train'],
                   val_bags=self.counts['val'], epoch_seconds=time.monotonic() - self.started)
        self.learn.lora_history.append(row)


class _Identifiers(Callback):
    def before_validate(self):
        self.slides, self.patients = [], []
        self.has_patients = False

    def after_pred(self):
        batch = self.xb[0]
        tiles = batch['tiles'] if isinstance(batch, dict) else batch[0]
        n = len(tiles)
        offset = len(self.slides)
        ids = batch.get('slide') if isinstance(batch, dict) else None
        patients = batch.get('patient') if isinstance(batch, dict) else None
        self.slides.extend(ids if ids is not None else [f'bag_{i:06d}' for i in range(offset, offset + n)])
        if patients is not None:
            self.has_patients = True
            self.patients.extend(patients)
        else:
            self.patients.extend([None] * n)
        if len(self.slides) != offset + n or len(self.patients) != offset + n:
            raise ValueError('slide and patient metadata must have one identifier per bag')


def build_learner(extractor, head, train_batches, val_batches, *, config,
                  loss_fn=None, forward_fn=None, metrics=None, first_block=None,
                  rank=8, alpha=16, dropout=0.05, adapt=True, device=None,
                  seed=None, outdir=None, lr_adapter=5e-5, lr_head=None, categories=None):
    """Prepare raw-tile loaders and the shared FastAI learner."""
    for source in (train_batches, val_batches):
        if source is not None and (isinstance(source, Iterator) or not hasattr(source, '__len__')):
            raise ValueError('FastAI requires sized, re-iterable training and validation loaders')
    if not len(train_batches) or (val_batches is not None and not len(val_batches)):
        raise ValueError('training and validation loaders cannot be empty')
    root = Path(outdir) if outdir is not None else None
    if root is not None and root.exists():
        raise FileExistsError(f'output directory already exists: {root}')
    config = copy.deepcopy(config)
    if config.epochs < 1:
        raise ValueError('epochs must be positive')
    if val_batches is None:
        config.save_monitor = 'train_loss'
        metrics = []
    if loss_fn is None:
        loss_kwargs = {}
        if config.is_classification() and config.weighted_loss:
            dataset = getattr(train_batches, 'dataset', None)
            targets = getattr(dataset, 'targets', None)
            if targets is None and hasattr(dataset, 'tensors'):
                targets = dataset.tensors[1]
            if targets is not None:
                targets = torch.as_tensor(targets)
                labels = targets.argmax(-1) if targets.ndim == 2 else targets
            else:
                labels = []
                for batch in train_batches:
                    targets = batch['targets'] if isinstance(batch, dict) else batch[1]
                    targets = torch.as_tensor(targets)
                    labels.append(targets.argmax(-1) if targets.ndim == 2 else targets)
                labels = torch.cat(labels)
            labels = labels.long()
            if torch.any(labels < 0):
                raise ValueError('class indices must be nonnegative')
            if categories is not None and torch.any(labels >= len(categories)):
                raise ValueError('class indices must fit categories')
            counts = labels.bincount(minlength=len(categories) if categories is not None else 0).float()
            weights = torch.where(counts > 0, counts.sum() / counts.clamp_min(1), 1.)
            loss_kwargs['weight'] = weights / weights.sum()
        loss_fn = config.loss_fn(**loss_kwargs)
    if not callable(loss_fn):
        raise ValueError('loss_fn must be callable')
    model = extractor.model
    device = torch.device(device or next(model.parameters()).device)
    first_block = max(0, len(model.blocks) - 8) if first_block is None else first_block
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    if adapt:
        init_lora(model, first_block=first_block, rank=rank, alpha=alpha, dropout=dropout)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        model.first_adapted = len(model.blocks)
    wrapped = LoRAModel(extractor, head, forward_fn, adapt).to(device)
    if not _splitter(wrapped):
        raise ValueError('the model has no trainable parameters')
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device=device, dtype=torch.float32)
    temporary = tempfile.TemporaryDirectory(prefix='slideflow-lora-') if root is None else None
    root = Path(temporary.name) if temporary else root
    root.mkdir(parents=True, exist_ok=True)
    train_dl = DataLoader(_Batches(train_batches), bs=None, device=device)
    val_dl = DataLoader(_Batches(val_batches if val_batches is not None else []), bs=None, device=device)
    train_dl.n_inp = val_dl.n_inp = 1
    dls = DataLoaders(train_dl, val_dl, device=device)
    if metrics is None:
        metrics = config.get_metrics()
        if config.is_classification():
            metrics = [_CategoricalMetric(metric) if getattr(metric, 'name', None) == 'roc_auc_score'
                       else metric for metric in metrics]
    learner = _fastai.build_learner_from_dls(
        config, dls, wrapped, loss_func=_Loss(loss_fn), metrics=metrics,
        path=root, splitter=_splitter, train_bn=False, wd=config.wd)
    head_lr = lr_head if lr_head is not None else config.lr
    learner.lora_lrs = ([lr_adapter] if any(p.requires_grad for p in model.parameters()) else [])
    if any(p.requires_grad for p in head.parameters()):
        learner.lora_lrs.append(head_lr)
    if head_lr is None:
        learner.lora_lrs = None
    learner.lora_config = config
    learner.lora_temporary = temporary
    learner.lora_settings = {
        'encoder': getattr(extractor, 'tag', type(model).__name__), 'head': type(head).__name__,
        'loss': getattr(loss_fn, '__name__', type(loss_fn).__name__),
        'adapt': adapt, 'seed': seed, 'epochs': config.epochs,
        'first_block': first_block if adapt else None, 'rank': rank if adapt else None,
        'alpha': alpha if adapt else None, 'dropout': dropout if adapt else None,
        'lr_adapter': lr_adapter if adapt else None, 'lr_head': head_lr,
    }
    return learner


def _validation_results(learner, outcomes, categories, prediction_fn):
    identities = _Identifiers()
    outputs, targets = learner.get_preds(reorder=False, act=lambda value: value, cbs=[identities])
    learner.lora_validation = {'outputs': _cpu(outputs), 'targets': _cpu(targets)}
    config = learner.lora_config
    if prediction_fn is not None:
        frame = prediction_fn(outputs, targets, identities.slides)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError('prediction_fn must return a dataframe with Slideflow outcome columns')
    else:
        if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise ValueError('structured outputs require prediction_fn for standard metric exports')
        outputs, targets = outputs.float().cpu(), targets.cpu()
        if outputs.ndim == 1:
            outputs = outputs[:, None]
        names = [outcomes] if isinstance(outcomes, str) else list(outcomes)
        frame = pd.DataFrame({'slide': identities.slides})
        if config.is_classification():
            if len(names) != 1:
                raise ValueError('classification needs one outcome name')
            if targets.ndim == 2:
                targets = targets.argmax(-1) if targets.shape[-1] > 1 else targets[:, 0]
            # a single logit represents the positive class
            scores = (torch.cat([1 - outputs.sigmoid(), outputs.sigmoid()], dim=1)
                      if outputs.shape[1] == 1 else outputs.softmax(-1))
            if categories is not None and len(categories) != scores.shape[1]:
                raise ValueError('categories must match the number of class scores')
            frame[f'{names[0]}-y_true'] = targets.numpy()
            for index in range(scores.shape[1]):
                frame[f'{names[0]}-y_pred{index}'] = scores[:, index].numpy()
        else:
            if targets.ndim == 1:
                targets = targets[:, None]
            if outputs.shape != targets.shape or outputs.shape[1] != len(names):
                raise ValueError('regression outcome names and target shape must match model outputs')
            for index, name in enumerate(names):
                frame[f'{name}-y_true'] = targets[:, index].numpy()
                frame[f'{name}-y_pred'] = outputs[:, index].numpy()
    if len(frame) != len(identities.slides) or 'slide' not in frame:
        raise ValueError('prediction_fn must preserve one identified row per validation bag')
    if identities.has_patients:
        frame['patient'] = identities.patients
    return frame


def train(extractor, head, train_batches, *, loss_fn=None, forward_fn=None,
          val_batches=None, config=None, metrics=None, callbacks=None,
          epochs=None, first_block=None, rank=8, alpha=16, dropout=0.05,
          lr_adapter=5e-5, lr_head=None, weight_decay=1e-5, adapt=True,
          device=None, seed=None, outdir=None, outcomes='outcome', categories=None,
          prediction_fn=None, return_learner=False):
    """Fit with the shared trainer and export the selected checkpoint's results."""
    if config is None:
        if loss_fn is None:
            raise ValueError('supply config or loss_fn')
        loss_name = 'mse' if isinstance(loss_fn, (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss)) else 'cross_entropy'
        config = mil_config(type(head), loss=loss_name, weighted_loss=False,
                            epochs=epochs if epochs is not None else 8,
                            lr=lr_head if lr_head is not None else 2e-4, wd=weight_decay,
                            fit_one_cycle=False)
        if metrics is None:
            metrics = []
    else:
        config = copy.deepcopy(config)
        if epochs is not None:
            config.epochs = epochs
    learner = build_learner(
        extractor, head, train_batches, val_batches, config=config,
        loss_fn=loss_fn, forward_fn=forward_fn, metrics=metrics,
        first_block=first_block, rank=rank, alpha=alpha, dropout=dropout,
        adapt=adapt, device=device, seed=seed, outdir=outdir,
        lr_adapter=lr_adapter, lr_head=lr_head, categories=categories)
    _fastai.train(learner, learner.lora_config,
                  callbacks=[_History()] + list(callbacks or []), lr=learner.lora_lrs)
    if outdir is not None:
        root = learner.path
        if adapt:
            torch.save(adapter_state_dict(extractor.model, first_block=learner.lora_settings['first_block']),
                       root / 'adapters.pt')
        torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, root / 'head.pt')
        params = _log_mil_params(learner.lora_config, outcomes, categories, None,
                                getattr(extractor.model, 'num_features', None),
                                getattr(learner, 'lora_output_shape', None))
        params.update(training_input='raw_tiles', lora=learner.lora_settings,
                      weights='head.pt', model_checkpoint='models/best_valid.pth')
        if val_batches is not None:
            frame = _validation_results(learner, outcomes, categories, prediction_fn)
            if isinstance(learner.lora_validation['outputs'], torch.Tensor):
                outputs = learner.lora_validation['outputs']
                params['output_shape'] = outputs.shape[-1] if outputs.ndim > 1 else 1
            frame.to_parquet(root / 'predictions.parquet')
            learner.lora_config.run_metrics(frame, level=learner.lora_config.aggregation_level, outdir=str(root))
            learner.lora_predictions = frame
            torch.save(learner.lora_validation, root / 'validation_outputs.pt')
        (root / 'mil_params.json').write_text(json.dumps(params, indent=2) + '\n')
        (root / 'history.json').write_text(json.dumps({**learner.lora_settings,
                                                    'history': learner.lora_history}, indent=2) + '\n')
    return learner if return_learner else learner.lora_history
