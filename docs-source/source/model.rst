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

``slideflow.model.lora`` trains query/value adapters in a timm ViT with any compatible PyTorch bag model. It has no dependency on nnMIL or a particular outcome. H-Optimus-0 and Mettle extractors support adapter loading; other encoders must have a compatible timm ViT structure with packed query/key/value projections.

.. code-block:: python

    import torch
    import slideflow as sf
    from slideflow.mil import mil_config
    from slideflow.model.lora import train_lora, predict_lora

    extractor = sf.build_feature_extractor(
        'hoptimus0', weights='/path/to/base/pytorch_model.bin')
    head = mil_config('attention_mil').build_model(1536, 2)
    history = train_lora(
        extractor, head, train_batches,
        loss_fn=torch.nn.CrossEntropyLoss(),
        val_batches=validation_batches,
        first_block=32, rank=8, alpha=16, seed=42,
        outdir='runs/fold0')
    predictions = predict_lora(extractor, head, validation_batches)
    logits = torch.cat(predictions['outputs'])

Choose ``nnmil`` or ``bistro.transformer`` instead when constructing the head; the trainer is unchanged. Model-specific configuration belongs in ``model_kwargs``. Binary, regression, and multi-task training use an appropriate caller-supplied loss; the trainer does not select an objective, rescale targets, or decode predictions.

Batches contain ``(tiles, targets)`` or ``(tiles, targets, lengths)``. Tiles are uint8 tensors shaped ``(bags, tiles, height, width, 3)``. Lengths mark valid tiles in padded bags. Dictionary batches use ``tiles``, ``targets`` and optional ``lengths`` keys. Targets can be tensors or nested dictionaries/tuples. Cross entropy requires integer class indices; floating-point targets are converted to float32. The extractor supplies normalization. Use re-iterable loaders for multiple epochs and supply patient-grouped splits and tile sampling outside the trainer.

By default, models with ``use_lens=True`` receive ``head(features, lengths)``; other models receive ``head(features)`` and require unpadded bags. To support another signature, metadata, or structured outputs, supply ``forward_fn(head, features, lengths, batch)``. The original batch is available for coordinates or auxiliary inputs; move those inputs to the feature device inside the callback. ``loss_fn(outputs, targets)`` must return a finite scalar tensor. Validation uses the same forward callback and loss. For example:

.. code-block:: python

    def forward_fn(head, features, lengths, batch):
        return head(features, coords=batch['coords'].to(features.device))

    def loss_fn(outputs, targets):
        return torch.nn.functional.cross_entropy(outputs[0], targets)

``predict_lora`` returns lists of detached outputs and targets, one entry per batch, preserving dictionaries and tuples. Apply task-specific softmax, sigmoid, scaling, or aggregation in the caller.

The encoder's base weights stay frozen; adapters and the prediction model update together. ``first_block`` defaults to the last eight blocks. A fresh extractor with ``adapt=False`` provides a frozen-encoder control. ``seed`` covers adapter initialization and training randomness; seed head construction and loader sampling separately.

Outputs are ``adapters.pt``, ``head.pt`` and ``history.json`` in a new directory. History records model and loss names, adapter settings, train/validation loss and bag counts. Preserve the caller's model configuration and target encoding alongside these outputs. Reload adapters using the extractor's ``lora`` argument with matching ``lora_first_block``, ``lora_rank`` and ``lora_alpha``; instantiate the same head and load ``head.pt``. nnMIL also provides ``NNMIL.from_checkpoint`` for current and legacy checkpoint names. Features extracted after adaptation can be used for ordinary saved-feature MIL training and architecture comparisons.

.. autofunction:: slideflow.model.lora.train_lora
.. autofunction:: slideflow.model.lora.predict_lora
