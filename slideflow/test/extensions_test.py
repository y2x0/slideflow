"""Verify opt-in registration leaves existing models and trainer functions intact."""

import os
import subprocess
import sys


def test_extension_import_does_not_replace_existing_behavior():
    script = r'''
import torch
import slideflow as sf
import slideflow.mil as mil
from slideflow.mil import _params, data, eval, utils
from slideflow.mil.train import _fastai
from slideflow.mil import _registry
from slideflow.model.extractors import _registry as extractors

modules = [mil, _params, data, eval, utils, _fastai]
originals = {module: dict(vars(module)) for module in modules}
models = dict(_registry._mil_models)
trainers = dict(_registry._mil_trainers)
encoders = dict(extractors._torch_extractors)
assert 'lora' not in models and 'nnmil' not in models
config = mil.mil_config('attention_mil', model_kwargs={'z_dim': 8, 'dropout_p': 0})
model = config.build_model(16, 2).eval()
bags = torch.randn(2, 4, 16)
lens = torch.tensor([2, 4])
before = model(bags, lens).detach()

import slideflow.mil.extensions

for module, previous in originals.items():
    for name, value in previous.items():
        assert vars(module)[name] is value, (module.__name__, name)
assert all(_registry._mil_models[k] is v for k, v in models.items())
assert _registry._mil_trainers == trainers
assert all(extractors._torch_extractors[k] is v for k, v in encoders.items())
assert set(_registry._mil_models) - set(models) == {'nnmil', 'lora'}
assert set(extractors._torch_extractors) - set(encoders) == {'mettle', 'hoptimus0_lora'}
assert type(mil.mil_config('attention_mil').model_config) is _params.MILModelConfig
torch.testing.assert_close(model(bags, lens), before, atol=0, rtol=0)
print('Existing registrations, functions, classes and predictions unchanged')
'''
    result = subprocess.run([sys.executable, '-c', script], env=dict(os.environ),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
