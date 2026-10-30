# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib.util
import json
import os
import sys

import pytest

from tokenspeed.runtime.utils.hf_transformers_utils import get_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def test_local_llama_config_keeps_checkpoint_values(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 512,
                "vocab_size": 256,
            }
        )
    )
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert config.hidden_size == 128
    assert config.num_hidden_layers == 2
    assert config.max_position_embeddings == 512
    assert config.architectures == ["LlamaForCausalLM"]


@pytest.mark.parametrize(
    "model_type,family",
    [
        ("minimax_m3_vl", "minimax_m3_vl"),
        ("nemotron_h", "nemotron_h"),
        ("nemotron_h_omni", "nemotron_h"),
        ("NemotronH_Nano_Omni_Reasoning_V3", "nemotron_h"),
    ],
)
def test_unavailable_model_family_has_dependency_error(tmp_path, model_type, family):
    dependency = f"transformers.models.{family}"
    if importlib.util.find_spec(dependency) is not None:
        pytest.skip("This Transformers installation supports the model family")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    with pytest.raises(ImportError, match=family):
        get_config(str(tmp_path), trust_remote_code=False)


@pytest.mark.parametrize(
    "architecture",
    ["LlamaForCausalLM", "Qwen3ForCausalLM", "DeepseekV3ForCausalLM"],
)
def test_model_registry_resolves_available_architecture(architecture):
    from tokenspeed.runtime.models.registry import ModelRegistry

    model_class, resolved = ModelRegistry.resolve_model_cls(architecture)
    assert resolved == architecture
    assert model_class.__name__ == architecture
