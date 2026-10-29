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

"""Check explicit TRT-LLM helper errors when its optional package is absent."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_KERNEL_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("module", "call", "message"),
    [
        (
            "gemm/trtllm.py",
            "module.dsv3_fused_a_gemm(x, x)",
            "Fused-A GEMM requires a compatible trtllm_kernel package",
        ),
        (
            "quantization/trtllm.py",
            "module._per_token_group_quant_8bit(x, 128, False)",
            "TRT-LLM FP8 quantization requires a compatible trtllm_kernel package",
        ),
    ],
)
def test_missing_trtllm_helper(module, call, message):
    # Load the leaf directly so unrelated optional backends do not participate.
    # Each interpreter has its own module cache and real kernel registry.
    source = _KERNEL_ROOT / "python/tokenspeed_kernel/ops" / module
    script = f"""
import importlib.util
import sys
import torch

sys.modules["trtllm_kernel"] = None
spec = importlib.util.spec_from_file_location("trtllm_helper_test", {str(source)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
x = torch.zeros((1, 128))
try:
    {call}
except RuntimeError as error:
    assert str(error) == {message!r}, str(error)
else:
    raise AssertionError("Missing TRT-LLM helper did not report its dependency")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_KERNEL_ROOT / "python"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout
