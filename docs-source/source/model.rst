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

LoRA is registered as ``mil_config('lora', ...)`` and uses the normal ``project.train_mil`` pipeline. The model contains a compatible timm ViT encoder and a selectable MIL aggregation head. There is no separate LoRA trainer.

.. code-block:: python

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

Choose ``head='attention_mil'`` with ``head_kwargs={'z_dim': 256}``, or ``head='bistro.transformer'`` with its constructor arguments. ``encoder='mettle'`` uses a local Mettle checkpoint. Encoders need compatible packed Q/K/V projections and a Slideflow transform/feature-width interface.

The bags directory contains one ``slide_name.pt`` tensor per slide: uint8 RGB tiles shaped ``(tiles, height, width, 3)``, before normalization. Existing cached tiles may be repacked; precomputed feature vectors cannot propagate encoder gradients. This interface reads tensor bags, not WSI/TFRecord streams. The existing bag loader reads complete files before sampling. Use small ``bag_size`` and ``batch_size`` settings; ``tile_batch_size`` chunks encoding and ``checkpoint_blocks=True`` checkpoints adapted transformer blocks.

Slide/patient grouping, annotation encoding, bag sampling, class weights, metrics, optimizer schedule and checkpoint selection all remain in the standard training path. The encoder is frozen except for its adapters. ``adapt=False`` provides a frozen-encoder control with the same head, data and trainer. The head uses bag lengths where supported; other heads receive trimmed bags. Head-specific trainer settings, such as nnMIL balanced sampling, are not implicitly copied to the wrapper. Multi-input or coordinate-dependent heads require a separate input contract.

Runs save the standard history, manifest, MIL parameters, predictions, metric plots and ``models/best_valid.pth``. That checkpoint contains the full encoder and head. ``sf.mil.load_model_weights(run_directory)`` restores the full raw-tile model; keep the base checkpoint available at its configured path. Patient predictions and metrics retain patient IDs; patient attention is not mapped onto individual slide heatmaps.

Use ``sf.mil.build_fastai_learner`` to customize training with ordinary FastAI callbacks. Regression uses ``loss='mse'``; other objectives follow normal custom MIL configurations. After best-model selection, ``learner.model.export_adapters('adapters.pt')`` and ``torch.save(learner.model.head.state_dict(), 'head.pt')`` create deployment weights. Reload adapters with matching first-block, rank and alpha settings into the same base extractor. nnMIL's ``NNMIL.from_checkpoint`` supports current and legacy head names.

The former ``train_lora``, ``build_lora_learner`` and ``predict_lora`` entry points have been replaced by ``train_mil``, ``build_fastai_learner`` and ``predict_mil``. Existing independent-loader callers must supply slide-named tile bags and standard Slideflow datasets.

.. autoclass:: slideflow.mil.models.LoRA
    :members: export_adapters
