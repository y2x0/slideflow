# Compare MIL models on shared features

nnMIL is a registered MIL model. LoRA is an independent encoder-training module that accepts a prediction model and loss. Both can be used for classification, regression, or another caller-defined task.

## Source map

- [nnMIL model and checkpoint loader](../../slideflow/mil/models/nnmil.py)
- [MIL registration](../../slideflow/mil/__init__.py) and [configuration](../../slideflow/mil/_params.py)
- [Generic LoRA training and prediction](../../slideflow/model/lora.py)
- [FastAI LoRA learner and result exports](../../slideflow/model/_lora_fastai.py)
- [Shared FastAI learner factory and trainer](../../slideflow/mil/train/_fastai.py)
- [Adapter insertion and loading](../../slideflow/model/extractors/_lora.py)
- [H-Optimus extractor](../../slideflow/model/extractors/hoptimus0.py)
- [Mettle extractor](../../slideflow/model/extractors/mettle.py)
- [Training tests with several heads](../../slideflow/test/lora_train_test.py)

## Compare saved-feature models

Prepare `train` and `val` datasets with a categorical `label` annotation and fixed, disjoint patient groups. Extract feature bags with the same encoder checkpoint, tile resolution, and sampling procedure. Use those same bags and splits for each architecture:

```python
from slideflow.mil import mil_config

for architecture in ['nnmil', 'attention_mil', 'bistro.transformer']:
    config = mil_config(architecture, lr=2e-4, epochs=40, bag_size=512)
    project.train_mil(
        config=config, outcomes='label',
        train_dataset=train, val_dataset=val,
        bags='/path/to/matched/features',
        outdir=f'runs/{architecture}',
    )
```

CLAM models are available through the optional `slideflow-gpl` package. Add `clam_sb` to a comparison after installing it. Configure model-specific arguments separately; architectures need not share the same hyperparameters. Select hyperparameters within training folds and report held-out results using the same patient grouping and metrics.

## Train adapters with a chosen model

```python
from slideflow.mil import mil_config
from slideflow.model.lora import train_lora

config = mil_config('attention_mil', lr=2e-4, epochs=8)
head = config.build_model(extractor.num_features, 2)
learner = train_lora(
    extractor, head, train_batches,
    config=config, val_batches=validation_batches,
    outcomes='label', categories=['negative', 'positive'],
    outdir='runs/adaptation', return_learner=True,
)
```

This runs through the existing Slideflow FastAI learner and trainer. The run saves the usual CSV history, best checkpoint, MIL configuration, validation predictions and metric plots, plus adapters and head weights from the selected checkpoint. Supply standard FastAI callbacks with `callbacks=[...]`, or use `build_lora_learner` to customize a Learner before fitting.

The loader yields uint8 tiles `(bags, tiles, height, width, 3)` and integer class targets. Swap in nnMIL, Bistro, or a custom PyTorch model without changing the trainer. Sized, re-iterable loaders keep their own sampling and grouping. Dictionary batches may include `slide` and `patient` identifiers for validation exports. Length-aware models accept padded bags; other signatures use `forward_fn`. Regression uses `mil_config(..., loss='mse')`; custom objectives supply `loss_fn`, compatible `metrics`, and a `prediction_fn` callback for the validation table. See the LoRA section in `docs-source/source/model.rst` for the complete contract.

After adapting an encoder, extract features once and benchmark several MIL models on those features. Keep encoder adaptation inside each training fold; validation and held-out labels must not influence adaptation. This example provides a comparison workflow and does not claim a measured performance improvement for any architecture.
