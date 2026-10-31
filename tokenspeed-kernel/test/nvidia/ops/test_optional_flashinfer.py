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

import pytest
import torch


def test_missing_moe_api_rejects_explicit_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tokenspeed_kernel.ops.moe import moe_plan
    from tokenspeed_kernel.selection import NoKernelFoundError
    from tokenspeed_kernel.thirdparty.flashinfer.availability import has_flashinfer_api

    if has_flashinfer_api("flashinfer", "ActivationType"):
        pytest.skip("Requires a wheel without the modern MoE API")
    with pytest.raises(NoKernelFoundError):
        moe_plan(
            "unquant",
            activation="silu",
            hidden=256,
            ispp=128,
            swiglu_form="standard",
            activation_clamped=False,
            expert_id_repeats=False,
            fast_math=False,
            combine_order="rank",
            features=frozenset(),
            solution="flashinfer_cutlass",
        )


def test_missing_gdn_api_rejects_explicit_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tokenspeed_kernel.ops.attention.gdn import gdn_decode_step
    from tokenspeed_kernel.selection import NoKernelFoundError
    from tokenspeed_kernel.thirdparty.flashinfer.availability import has_flashinfer_api

    if has_flashinfer_api(
        "flashinfer.gdn_decode", "gated_delta_rule_decode_pretranspose"
    ):
        pytest.skip("Requires a wheel without the GDN decode API")
    q = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16, device="cuda")
    gate = torch.zeros(1, 1, 1, dtype=q.dtype, device=q.device)
    scalar = torch.zeros(1, device=q.device)
    state = torch.zeros(1, 1, 128, 128, dtype=torch.float32, device=q.device)
    indices = torch.zeros(1, dtype=torch.int32, device=q.device)
    with pytest.raises(NoKernelFoundError):
        gdn_decode_step(
            q,
            q,
            q,
            A_log=scalar,
            a=gate,
            dt_bias=scalar,
            b=gate,
            initial_state=state,
            initial_state_indices=indices,
            solution="flashinfer",
        )
