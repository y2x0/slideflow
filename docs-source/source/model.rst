.. currentmodule:: slideflow.model

slideflow.model
===============

This module provides the :class:`ModelParams` class to organize model and training
parameters/hyperparameters and assist with model building, as well as the :class:`Trainer` class that
executes model training and evaluation. :class:`RegressionTrainer` and :class:`SurvivalTrainer`
are extensions of this class, supporting regression and Cox Proportional Hazards outcomes, respectively. The function
:func:`build_trainer` can choose and return the correct model instance based on the provided
hyperparameters.

.. note::
    In order to support both Tensorflow and PyTorch backends, the :mod:`slideflow.model` module will import either
    :mod:`slideflow.model.tensorflow` or :mod:`slideflow.model.torch` according to the currently active backend,
    indicated by the environmental variable ``SF_BACKEND``.

See :ref:`training` for a detailed look at how to train models.

Trainer
*******
.. autoclass:: Trainer
.. autofunction:: slideflow.model.Trainer.load
.. autofunction:: slideflow.model.Trainer.evaluate
.. autofunction:: slideflow.model.Trainer.predict
.. autofunction:: slideflow.model.Trainer.train

RegressionTrainer
*****************
.. autoclass:: RegressionTrainer

SurvivalTrainer
***************
.. autoclass:: SurvivalTrainer

Features
********
.. autoclass:: Features
.. autofunction:: slideflow.model.Features.from_model
.. autofunction:: slideflow.model.Features.__call__

Other functions
***************
.. autofunction:: build_trainer
.. autofunction:: build_feature_extractor
.. autofunction:: list_extractors
.. autofunction:: load
.. autofunction:: is_tensorflow_model
.. autofunction:: is_tensorflow_tensor
.. autofunction:: is_torch_model
.. autofunction:: is_torch_tensor
.. autofunction:: read_hp_sweep
.. autofunction:: rebuild_extractor

.. _lora_training:

LoRA training
*************

``slideflow.model.lora`` uses Slideflow's existing FastAI MIL trainer for raw-tile encoder adaptation. A wrapper combines a compatible timm ViT encoder and any PyTorch bag model; the shared learner factory supplies metrics and the shared trainer supplies scheduling, CSV logging and best-model selection. LoRA has no dependency on nnMIL or a particular outcome.

.. code-block:: python

    import slideflow as sf
    from slideflow.mil import mil_config
    from slideflow.model.lora import train_lora

    extractor = sf.build_feature_extractor(
        'hoptimus0', weights='/path/to/base/pytorch_model.bin')
    config = mil_config('attention_mil', lr=2e-4, epochs=8)
    head = config.build_model(extractor.num_features, 2)
    learner = train_lora(
        extractor, head, train_batches,
        config=config, val_batches=validation_batches,
        outcomes='label', categories=['negative', 'positive'],
        first_block=32, rank=8, alpha=16,
        lr_adapter=5e-5, seed=42,
        outdir='runs/fold0', return_learner=True)

Choose ``nnmil`` or ``bistro.transformer`` when constructing the configuration to use a different head. Model-specific arguments belong in ``model_kwargs``. ``config`` controls loss, weighted classification loss, metrics, epochs, weight decay, scheduling and checkpoint monitor. Regression uses ``mil_config(..., loss='mse')`` with floating-point targets. ``loss_fn`` overrides the configured loss; ``metrics`` overrides configured metrics; ``callbacks`` adds standard FastAI callbacks. Explicit ``epochs`` and ``lr_head`` override those settings. Otherwise the head uses ``config.lr``, with a separate adapter learning rate. When ``config.lr`` is None, the shared trainer uses FastAI's learning-rate finder.

A run saves ``history.csv``, ``models/best_valid.pth``, ``mil_params.json`` and ``predictions.parquet``. Validation results use Slideflow's ordinary classification/regression metric functions and plots. The best-model callback restores the selected encoder/head checkpoint before validation export. ``adapters.pt`` and ``head.pt`` are exported from that same selected model, and ``history.json`` records adapter settings and epoch metrics. The standard FastAI checkpoint contains the full encoder and head; adapter-only weights remain available for deployment. ``sf.mil.load_model_weights(run_directory)`` loads the exported head, which must be used with features from its matching adapted encoder.

For customization before fitting, call ``build_lora_learner(extractor, head, train_batches, validation_batches, config=config, ...)``. It returns a normal FastAI Learner without training. Standard Learner methods and callbacks can then be used (use ``get_preds(reorder=False)`` with these pre-batched iterable loaders); the high-level ``train_lora`` call handles the complete training and result-export sequence. For compatibility, ``train_lora`` returns a history list unless ``return_learner=True``. The earlier loss-only call is also supported and now uses FastAI rather than a separate optimizer loop.

Batches contain ``(tiles, targets)`` or ``(tiles, targets, lengths)``. Tiles are uint8 tensors shaped ``(bags, tiles, height, width, 3)``. Lengths mark valid tiles in padded bags. Dictionary batches use ``tiles``, ``targets`` and optional ``lengths`` keys; ``slide`` and ``patient`` can supply one identifier per bag for saved validation predictions. Without identifiers, deterministic row identifiers are assigned. Targets can be tensors or nested dictionaries/tuples. Cross entropy accepts integer class indices or one-hot targets; floating-point targets are converted to float32. The ROC metric handles class indices with the same target encoding as Slideflow's saved-feature loaders. Categories must match the target indices; supplying all categories also covers a class absent from training when weighting the loss.

The extractor supplies normalization. Loaders must be sized and re-iterable; their sampling and grouping are preserved. Supply patient-grouped splits and tile sampling outside the trainer. ``config.aggregation_level`` describes the caller's bag grouping; this API does not regroup raw tile loaders. Validation uses the same forward callback and loss as training. Without a validation loader, checkpoint selection uses training loss and no held-out validation artifacts are produced.

Models with ``use_lens=True`` receive ``head(features, lengths)``; other models receive ``head(features)`` and require unpadded bags. To support another signature, metadata, or structured outputs, supply ``forward_fn(head, features, lengths, batch)``. The batch is available for coordinates or auxiliary inputs. ``loss_fn(outputs, targets)`` must return a finite scalar tensor. Structured objectives also supply compatible ``metrics``; for standard validation exports, ``prediction_fn(outputs, targets, slide_ids)`` returns a DataFrame with Slideflow outcome columns. Use this callback for task-specific prediction decoding.

``predict_lora`` remains a raw prediction helper: it returns lists of detached outputs and targets, one entry per batch, preserving dictionaries and tuples. Apply task-specific decoding in the caller. The saved validation table uses standard classification probabilities or continuous regression outputs.

Base encoder weights stay frozen; adapters and the head update together. ``first_block`` defaults to the last eight blocks. A fresh extractor with ``adapt=False`` provides a frozen-encoder control through the same training path. ``seed`` covers adapter initialization and training randomness; seed head construction and loader sampling separately.

H-Optimus-0 and Mettle extractors support adapter loading. Other encoders require a compatible timm ViT structure with packed query/key/value projections. Reload adapters with matching ``lora_first_block``, ``lora_rank`` and ``lora_alpha`` settings; preserve the model configuration and target encoding. nnMIL also provides ``NNMIL.from_checkpoint`` for current and legacy parameter names. Features extracted after adaptation can be used for ordinary saved-feature MIL training and architecture comparisons.

.. autofunction:: slideflow.model.lora.build_lora_learner
.. autofunction:: slideflow.model.lora.train_lora
.. autofunction:: slideflow.model.lora.predict_lora
