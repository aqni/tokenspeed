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

"""Bitwise head exchange and projection fallback with optional Lamport JIT."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _payload(rank, rows, size, device, phase):
    # Cover all BF16 bits, including NaN payloads and both zero signs.
    bits = torch.arange(rows * size * 32, device=device, dtype=torch.int64)
    return (
        ((bits * 31 + rank * 8191 + phase) % 65536)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(rows, size * 2, 16)
    )


@torch.no_grad()
def _worker(rank, size, rendezvous):
    from tokenspeed_kernel.ops.communication.cuda import (
        TokenSpeedA2ALamportState,
        tokenspeed_a2a_lamport_supported,
    )

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.projection import ProjectionSpec
    from tokenspeed.runtime.distributed.comm_ops import (
        all_to_all_head_scatter,
        all_to_all_transpose,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=180),
        device_id=device,
    )
    group = tuple(range(size))
    pg_manager.register_process_group("nccl", group, dist.group.WORLD)
    backend = AutoBackend()
    supported = tokenspeed_a2a_lamport_supported()
    assert (
        backend.prepare_shard_exchange(group, size * 32, 257, torch.bfloat16)
        == supported
    )
    if not supported:
        with pytest.raises(RuntimeError, match="CUDADeviceGuard"):
            TokenSpeedA2ALamportState(dist.group.WORLD, 257, size * 32, device, 1)

    for rows in (1, 17, 257):
        inputs = _payload(rank, rows, size, device, 0)
        scattered_out = torch.empty(
            rows * size, 2, 16, dtype=inputs.dtype, device=device
        )
        restored_out = torch.empty_like(inputs.reshape(rows, -1))

        def exchange():
            scattered = all_to_all_head_scatter(inputs, group, [rows] * size, backend)
            restored = all_to_all_transpose(
                scattered.flatten(1), group, [rows] * size, backend
            )
            scattered_out.copy_(scattered)
            restored_out.copy_(restored)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                exchange()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            exchange()
        for phase in (0, 1, 32768, 65535):
            inputs.copy_(_payload(rank, rows, size, device, phase))
            graph.replay()
            expected = torch.cat(
                [
                    _payload(source, rows, size, device, phase)[
                        :, 2 * rank : 2 * rank + 2
                    ]
                    for source in range(size)
                ]
            )
            assert torch.equal(
                scattered_out.view(torch.int16), expected.view(torch.int16)
            )
            assert torch.equal(
                restored_out.view(torch.int16), inputs.flatten(1).view(torch.int16)
            )
        del graph

    if not supported:
        spec = ProjectionSpec(
            group, "row", size * 32, size * 32, 257, torch.bfloat16, device
        )
        workspace = backend.prepare_projection(spec)
        inputs = _payload(rank, 17, size, device, 1).flatten(1)
        scattered, scales = backend.projection_all_to_all(
            inputs, 17, False, False, None, workspace
        )
        expected = torch.cat(
            [
                _payload(source, 17, size, device, 1)[
                    :, 2 * rank : 2 * rank + 2
                ].flatten(1)
                for source in range(size)
            ]
        )
        assert scales is None
        assert torch.equal(scattered.view(torch.int16), expected.view(torch.int16))
        restored = torch.empty_like(inputs)
        # Projection outputs are borrowed scratch; a following GEMM normally
        # produces separate storage before the inverse exchange.
        backend.projection_all_to_all(
            scattered.clone(), 17, True, False, restored, workspace
        )
        assert torch.equal(restored.view(torch.int16), inputs.view(torch.int16))
        workspace.close()
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_optional_lamport_exchange(size, tmp_path):
    if torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} CUDA GPUs on one host")
    mp.spawn(
        _worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=size,
        join=True,
    )
