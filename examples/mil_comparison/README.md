# Compare MIL models on shared features

nnMIL and LoRA are registered MIL models. Both use `mil_config(...)` and `project.train_mil(...)`. LoRA adds encoder adaptation around a selectable aggregation head.

## Source map

- [nnMIL model and checkpoint loader](../../slideflow/mil/models/nnmil.py)
- [MIL registration](../../slideflow/mil/__init__.py) and [configuration](../../slideflow/mil/_params.py)
- [LoRA model](../../slideflow/mil/models/lora.py)
- [Encoder forward utility](../../slideflow/model/lora.py)
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

config = mil_config(
    'lora',
    aggregation_level='patient',
    lr=5e-5, epochs=8, batch_size=2, bag_size=32,
    model_kwargs={
        'encoder': 'hoptimus0',
        'encoder_kwargs': {'weights': '/path/to/base/pytorch_model.bin'},
        'head': 'nnmil',  # or 'attention_mil', 'bistro.transformer'
        'head_kwargs': {'hidden_dim': 256},
        'first_block': 32, 'rank': 8, 'alpha': 16,
        'tile_batch_size': 16,
    },
)
learner = project.train_mil(
    config=config, outcomes='label',
    train_dataset=train, val_dataset=val,
    bags='/path/to/rgb_tile_bags',
    outdir='runs/adaptation',
)

```

For AMIL, use `head='attention_mil'` and `head_kwargs={'z_dim': 256}`. For Bistro, use `head='bistro.transformer'` and its own constructor arguments. The encoder can also be `mettle` with a local checkpoint. Compatible encoders must expose a timm ViT with packed Q/K/V projections and the Slideflow transform/feature-width interface. Multi-input heads and coordinate-dependent heads need a separate input contract.

Each bag is a `.pt` tensor named for its slide, containing **uint8 RGB tiles shaped `(tiles, height, width, 3)`**, before normalization. Pack existing decoded/cached tiles with `torch.save(tiles, 'slide_name.pt')`; LoRA cannot train the encoder from saved embeddings. This interface reads tensor bags; it does not directly stream WSI files or TFRecords. All slides in a bag directory must use the same tile resolution and channel layout. Keep coordinates in matching tile order if producing slide heatmaps.

The ordinary `train_mil` path resolves annotations and slide names, groups bags by slide or patient, builds the usual dataloaders, and constructs `LoRA` through `config.build_model`. Loss, class weighting, sampling, batch size, epochs, learning rate, schedule, validation and checkpoint monitor remain standard trainer settings. The head receives encoded features; length-aware heads mask padding, and other heads receive individually trimmed bags. Changing `head` does not choose a separate trainer or sampling policy. Head-specific config options such as nnMIL's `balanced_batches` are not inherited by the LoRA wrapper.

Start with small raw-tile bags and batches because encoder training uses more memory than saved-feature training. `tile_batch_size` chunks the encoder forward pass; `checkpoint_blocks=True` reduces activations stored for backward. The shared bag loader still reads complete slide tensor files before sampling. Set `adapt=False` inside `model_kwargs` for a frozen-encoder control using exactly the same pipeline and head. Keep patient splits, preprocessing, seeds, sampler, learning rate and schedule fixed for that comparison.

Runs save the ordinary `history.csv`, `models/best_valid.pth`, `mil_params.json`, `slide_manifest.csv`, `predictions.parquet` and metric plots. The checkpoint contains the encoder, adapters and head. `sf.mil.load_model_weights(run_directory)` reconstructs the full model for raw-tile predictions; retain the base checkpoint at its configured path. Patient-level metrics and predictions retain patient identifiers; patient attention is not exported as slide heatmaps.

For customization, use `sf.mil.build_fastai_learner` and standard FastAI methods or callbacks. For adapter-only deployment, call `learner.model.export_adapters('adapters.pt')` and `torch.save(learner.model.head.state_dict(), 'head.pt')` after the best checkpoint has been restored. Load adapters into the matching base extractor with the same first-block, rank and alpha settings. There is no separate `train_lora` or LoRA trainer.

Regression uses `loss='mse'` in `mil_config`. Other objectives follow the same custom MIL configuration mechanism as ordinary MIL. Adaptation must stay inside each training fold. This example is a software workflow, not a measured performance comparison.
