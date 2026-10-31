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

"""Unsupported optional FlashInfer paths fail at selection, not family import."""

import importlib.metadata as metadata

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from tokenspeed_kernel.ops.tuning import (
    autotune,
    is_autotuning,
    is_warming_up,
    kernel_warmup,
    load_autotune_cache,
    set_autotune_process_group,
)

pytestmark = pytest.mark.skipif(
    not metadata.version("flashinfer-python").startswith("0.5."),
    reason="Exercise the real legacy FlashInfer backend",
)


def test_legacy_flashinfer_keeps_normal_gpu_warmup():
    value = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=value.dtype)
    set_autotune_process_group(None)
    assert not load_autotune_cache(None, None, 0)
    with kernel_warmup(enabled=True):
        with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
            assert is_warming_up()
            assert not is_autotuning()
            actual = flashinfer.norm.rmsnorm(value, weight, eps=1e-6)
    torch.cuda.synchronize()
    expected = (
        value.float()
        * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(value.dtype)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
    assert not is_warming_up()
    assert not is_autotuning()


@pytest.mark.parametrize("buckets,round_up", [((8, 16), None), (None, True)])
def test_legacy_flashinfer_rejects_explicit_new_tuning_options(buckets, round_up):
    with pytest.raises(RuntimeError, match="compatible FlashInfer"):
        with autotune(tune_mode=True, tuning_buckets=buckets, round_up=round_up):
            pytest.fail("Unsupported options must not reach the wrapped operation")
