"""Model configuration hooks for the optional nnMIL and LoRA modules."""

import numpy as np
import torch
from slideflow.mil._params import MILModelConfig, TrainerConfig
from slideflow.mil.data import MapDataset, EncodedDataset
from ._extension_data import InstanceBagDataset, StratifiedShuffle, TileBagDataset, resolve_tile_bags


class ExtensionModelConfig(MILModelConfig):
    bag_dataset = InstanceBagDataset

    def __init__(self, model, *, num_workers=None, **kwargs):
        self.num_workers = num_workers
        super().__init__(model, **kwargs)

    def verify_trainer(self, trainer):
        self._aggregation_level = trainer.aggregation_level
        if self._aggregation_level == 'patient' and not self.is_classification():
            raise ValueError('Patient regression is unsupported by the unchanged Slideflow evaluation path; '
                             'use slide aggregation for regression')

    def to_dict(self):
        params = super().to_dict()
        params['aggregation_level'] = getattr(self, '_aggregation_level', 'slide')
        return params

    def build_model(self, n_in, n_out, **kwargs):
        model = super().build_model(n_in, n_out, **kwargs)
        model.uq_apply_softmax = self.is_classification()
        return model

    def _build_dataloader(self, bags, targets, encoder, *, dataset_kwargs=None,
                          dataloader_kwargs=None):
        from fastai.vision.all import DataLoader

        data_options = dict(dataset_kwargs or {})
        use_lens = data_options.pop('use_lens', self.use_lens)
        data_options.setdefault('dtype', getattr(self.model_fn, 'input_dtype', torch.float32))
        loader_options = dict(dataloader_kwargs or {})
        loader_options.pop('shufle', None)
        if self.num_workers is not None:
            loader_options['num_workers'] = self.num_workers
        if loader_options.get('num_workers') == 0:
            loader_options['persistent_workers'] = False

        def combine(bag, target):
            features, lengths = bag
            target = target if encoder is None else target.squeeze()
            return (features, lengths, target) if use_lens else (features, target)

        dataset = MapDataset(combine, self.bag_dataset(bags, **data_options),
                             EncodedDataset(encoder, targets))
        dataset.encoder = encoder
        return DataLoader(dataset, **loader_options)

    def predict(self, model, bags, attention=False, **kwargs):
        requested_attention = attention
        if getattr(self, '_aggregation_level', 'slide') == 'patient':
            attention = False
        if kwargs.get('uq'):
            kwargs['apply_softmax'] = False
        result = super().predict(model, bags, attention=attention, **kwargs)
        if requested_attention and not attention:
            return (result[0], [], *result[2:])
        return result

    def batched_predict(self, model, loaded_bags, **kwargs):
        if kwargs.get('uq'):
            kwargs['apply_softmax'] = False
            forward = dict(kwargs.get('forward_kwargs') or {})
            forward.setdefault('uq_softmax', self.is_classification())
            kwargs['forward_kwargs'] = forward
        return super().batched_predict(model, loaded_bags, **kwargs)

    def run_metrics(self, df, level='slide', outdir=None):
        level = getattr(self, '_aggregation_level', level)
        if level == 'patient':
            df = df.rename(columns={c: 'patient' for c in df if c.endswith('-patient')})
        return super().run_metrics(df, level=level, outdir=outdir)


class LoRAModelConfig(ExtensionModelConfig):
    bag_dataset = TileBagDataset

    def to_dict(self):
        params = super().to_dict()
        options = dict(params['model_kwargs'] or {})
        if options.pop('adapters', None) is not None:
            options['use_adapters'] = True
        options.pop('head_weights', None)
        params['model_kwargs'] = options
        return params

    def predict(self, model, bags, attention=False, **kwargs):
        dataset = TileBagDataset(bags, dtype=torch.uint8)
        loaded = (dataset[i][0] for i in range(len(dataset)))
        return super().predict(model, loaded, attention=attention, **kwargs)


class LoRAConfig(TrainerConfig):
    """Resolve tile paths, then use the standard MIL trainer and aggregation."""

    def train(self, train_dataset, val_dataset, outcomes, bags, **kwargs):
        return super().train(train_dataset, val_dataset, outcomes,
                             resolve_tile_bags(bags), **kwargs)

    def eval(self, model, dataset, outcomes, bags, **kwargs):
        return super().eval(model, dataset, outcomes, resolve_tile_bags(bags), **kwargs)


class NNMILModelConfig(ExtensionModelConfig):
    def __init__(self, model='nnmil', *, balanced_batches=True, n_strata=4, **kwargs):
        if not isinstance(n_strata, int) or n_strata < 1:
            raise ValueError('n_strata must be a positive integer')
        self.balanced_batches = balanced_batches
        self.n_strata = n_strata
        super().__init__(model, **kwargs)

    def _strata(self, targets):
        targets = np.asarray(targets)
        if self.is_classification():
            return np.unique(targets.reshape(len(targets), -1)[:, 0], return_inverse=True)[1]
        values = targets.reshape(len(targets), -1)[:, 0].astype(float)
        if not np.isfinite(values).all():
            raise ValueError('regression strata require finite targets')
        edges = np.quantile(values, np.linspace(0, 1, self.n_strata + 1)[1:-1])
        return np.digitize(values, edges)

    def _build_dataloader(self, bags, targets, encoder, *, dataset_kwargs=None,
                          dataloader_kwargs=None):
        loader_options = dict(dataloader_kwargs or {})
        if self.balanced_batches and loader_options.get('shuffle'):
            loader_options['shuffle_fn'] = StratifiedShuffle(self._strata(targets))
        return super()._build_dataloader(bags, targets, encoder, dataset_kwargs=dataset_kwargs,
                                         dataloader_kwargs=loader_options)
