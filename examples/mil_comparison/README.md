# Optional nnMIL, LoRA and Mettle modules

These additions are opt-in. Existing Slideflow files, README, configuration, models and trainers are unchanged from the pre-module baseline. No existing functions or registry entries are replaced at import time.

```python
import slideflow as sf
import slideflow.mil.extensions
from slideflow.mil import mil_config
```

The import registers two new MIL names (`nnmil`, `lora`) and two new extractor names (`mettle`, `hoptimus0_lora`). Standard `hoptimus0` is unchanged; `hoptimus0_lora` adds local checkpoint construction and adapter loading. Import the extension before reconstructing saved extension models as well.

## Train LoRA with a selected head

```python
config = mil_config(
    'lora',
    aggregation_level='patient',
    lr=5e-5, epochs=8, batch_size=2, bag_size=32,
    model_kwargs={
        'encoder': 'hoptimus0_lora',
        'encoder_kwargs': {'weights': '/path/to/base/pytorch_model.bin'},
        'head': 'nnmil',
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

Use `head='attention_mil'` with `head_kwargs={'z_dim': 256}` for AMIL, `head='bistro.transformer'` with its constructor arguments, or `head='transmil'` with `head_kwargs={}`. TransMIL requires the optional `nystrom-attention` package (`python -m pip install nystrom-attention`). Mettle uses `encoder='mettle'` and a local checkpoint. Heads operate on features from the encoder. Length-aware heads receive bag lengths; other heads receive individually trimmed bags. Multimodal and coordinate-dependent heads require a different input contract.

The unchanged Bistro implementation requires `dim == mlp_dim == heads * dim_head`. Its defaults satisfy this; a smaller valid example is `{'dim': 128, 'mlp_dim': 128, 'heads': 4, 'dim_head': 32, 'depth': 1}`.

| Head | Slide/patient classification | Slide regression | Head UQ |
| --- | --- | --- | --- |
| `nnmil` | Tested | Tested | Tested |
| `attention_mil` | Tested | Tested | Tested |
| `bistro.transformer` | Tested | Tested | Unsupported |
| `transmil` | Tested | Tested | Unsupported |

These checks use a tiny encoder and synthetic RGB bags on CPU. They cover the ordinary trainer, frozen controls, adapter/head gradients with and without activation checkpointing, bag padding, and classification checkpoint reload. TransMIL was tested with `nystrom-attention==0.0.14`. Other registered heads are not covered by this matrix. nnMIL can also be imported directly from `slideflow.mil.models.nnmil` without importing the LoRA modules.

A bounded NVIDIA H200 check on 8 October 2026 also passed all four heads with local pretrained H-Optimus and Mettle encoders: eight LoRA combinations, followed by standalone nnMIL fits on embeddings from each encoder. Each fit used six training and four held-out patients, four cached tiles per patient, and one epoch. The tests verified unchanged frozen encoder weights, changed adapters and heads, finite losses, patient grouping, and identical predictions after checkpoint reload. The 61-test software suite also passed in the cluster environment. An initial Bistro test configuration had inconsistent dimensions; the corrected configuration above passed. This establishes execution on real encoders and cached tiles, not cohort accuracy or full-scale memory requirements.

The normal `train_mil` code resolves annotation labels and slide/patient grouping, then the normal FastAI builder calls the registered model configuration's dataloader hook and `config.build_model`. The added configuration supplies an image-aware bag dataset only for the new models. The existing trainer still controls losses, class weighting, learning rate, epochs, scheduling, checkpoint selection and result exports. There is no separate LoRA trainer or replacement training function. Choosing a head does not inherit that head's model-configuration sampler settings.

The model forward is `RGB tiles -> encoder with adapters -> selected MIL head -> predictions`. The ordinary MIL loss trains the adapters and head jointly; the base encoder stays frozen. Adapter insertion, tile preprocessing and encoder activation checkpointing belong to the model. The additional dataset handles image-shaped padding through the existing configuration hook, while slide/patient grouping stays in `train_mil`.

Each bag is a slide-named `.pt` uint8 RGB tensor shaped `(tiles, height, width, 3)`, before normalization. Existing decoded tiles can be packed with `torch.save(tiles, 'slide_name.pt')`. Saved embeddings cannot train encoder adapters. This interface does not stream WSI/TFRecord files; the added dataset loads complete slide tensor files before sampling. Use small bags/batches. `tile_batch_size` chunks encoder forwards and `checkpoint_blocks=True` reduces stored activations. Set `adapt=False` in `model_kwargs` for a frozen-encoder control with the same data, head and trainer. Keep patient-disjoint splits fixed across comparisons.

For an exactly matched initial head in a LoRA/control comparison, copy the same head state into both models before fitting with the ordinary `sf.mil.build_fastai_learner`. A shared global seed alone does not guarantee identical head initialization: constructing adapters consumes random draws before the head is built. Keep bag order/sampling, trainer settings and encoder preprocessing matched as well.

`num_workers=0` may be passed to `mil_config` for local debugging of the added models; omitted, the extension retains the trainer's worker counts. Sampling and batching otherwise follow the usual trainer settings. LoRA uses uniform shuffled batches; standalone nnMIL optionally uses the added balanced sampler.

## Train nnMIL on existing feature bags

```python
config = mil_config('nnmil', lr=2e-4, epochs=40, bag_size=512,
                    balanced_batches=True, model_kwargs={'hidden_dim': 256})
learner = project.train_mil(
    config=config, outcomes='label',
    train_dataset=train, val_dataset=val,
    bags='/path/to/features', outdir='runs/nnmil',
)
```

Existing AMIL, TransMIL, Bistro and extractor registrations retain their original behavior. The added nnMIL configuration's sampling and data handling do not change those models.

## Results and limits

Runs use the unchanged trainer's history, manifests, MIL parameters, predictions, best checkpoint and metrics. `sf.mil.load_model_weights(run_directory)` restores the standard best checkpoint after the extension import. LoRA checkpoints contain the full encoder and head; keep the configured base checkpoint available when reconstructing the model.

The restored Slideflow code cannot format patient-level regression targets correctly during evaluation. The added configurations reject patient-level regression explicitly. Slide-level regression and slide/patient classification are supported. No upstream regression fix or split-checking behavior is installed. The caller must provide patient-disjoint train/validation datasets, as for ordinary MIL.

Patient-level attention from the added models is suppressed because the unchanged exporter assumes slide names; do not request patient attention heatmaps. Slide attention requires coordinates matching the tile order. These limits are retained rather than modifying shared Slideflow code.

For TransMIL and Bistro, attention requests must use unpadded bags one at a time, as the normal prediction path does. Direct attention requests with external padding raise an error; ordinary padded training and prediction remain supported. Attention outputs retain the selected head's native meaning; TransMIL exposes per-tile latent channels rather than normalized scalar attention weights.

The generic module does not package a complete study-specific training protocol. Joint classification/regression targets, separate adapter/head learning rates, cohort splits and a later feature-extraction/head-refit stage require their own configuration or workflow. Full-scale resource validation and cohort performance evaluation remain separate from the bounded execution checks. Encoder support is limited to compatible timm ViTs with packed Q/K/V projections.

Run the extension tests from the checkout with:

```sh
SF_BACKEND=torch SF_SLIDE_BACKEND=libvips OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q slideflow/test/extensions_test.py slideflow/test/lora_fastai_test.py \
slideflow/test/lora_train_test.py slideflow/test/nnmil_test.py slideflow/test/nnmil_integration_test.py
```

TransMIL cases are skipped when its optional dependency is absent; install it to run the full matrix.

After best-model selection, optional deployment exports are:

```python
import torch
learner.model.export_adapters('adapters.pt')
torch.save(learner.model.head.state_dict(), 'head.pt')
```

Reload adapters with the same base encoder, first block, rank and alpha. The additive `hoptimus0_lora` and `mettle` extractors accept those adapter parameters. For custom fitting, use the ordinary `sf.mil.build_fastai_learner` and FastAI callbacks. This is a software integration, not a measured cohort performance result.

## Added source files

- `slideflow/mil/extensions.py`: opt-in registration only.
- `slideflow/mil/_extension_config.py`: configuration hooks scoped to the added models.
- `slideflow/mil/_extension_data.py`: instance-shaped bags and optional nnMIL sampling.
- `slideflow/mil/models/{nnmil,lora}.py`: model classes.
- `slideflow/model/lora.py` and `extractors/_lora.py`: encoder forward and adapter utilities.
- `slideflow/model/extractors/{mettle,hoptimus0_lora}.py`: additional extractors.
- New tests in `slideflow/test/`: training, checkpoint, padding, extractor and import-isolation checks.
