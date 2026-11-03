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

"""Pinned allocator defaults follow the native allocator's capabilities."""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.utils import launcher  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
SETTING = "pinned_max_round_threshold_mb:1024"


@pytest.fixture(autouse=True)
def clear_allocator_environment(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("key", KEYS)
def test_default_preserves_existing_options(monkeypatch, supported, key):
    # CPU CI need not provide a CUDA/ROCm allocator implementation.
    monkeypatch.setattr(launcher, "_supports_allocator_setting", lambda _: supported)
    monkeypatch.setenv(key, "expandable_segments:False,")
    launcher.set_pinned_allocator_default()
    expected = (
        f"expandable_segments:False,{SETTING}"
        if supported
        else "expandable_segments:False,"
    )
    assert os.environ[key] == expected
    assert all(other not in os.environ for other in KEYS if other != key)


@pytest.mark.parametrize("supported", [True, False])
def test_default_with_no_existing_variable(monkeypatch, supported):
    monkeypatch.setattr(launcher, "_supports_allocator_setting", lambda _: supported)
    launcher.set_pinned_allocator_default()
    assert os.environ.get("PYTORCH_ALLOC_CONF") == (SETTING if supported else None)


def test_explicit_threshold_and_variable_precedence(monkeypatch):
    explicit = "pinned_max_round_threshold_mb:256"
    monkeypatch.setenv(KEYS[0], explicit)
    monkeypatch.setenv(KEYS[1], "expandable_segments:False")
    monkeypatch.setenv(KEYS[2], "expandable_segments:True")
    before = {key: os.environ[key] for key in KEYS}
    launcher.set_pinned_allocator_default()
    assert {key: os.environ[key] for key in KEYS} == before


def test_probe_does_not_hide_other_failures(monkeypatch):
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "unrelated import failure"
        ),
    )
    with pytest.raises(RuntimeError, match="unrelated import failure"):
        launcher.set_pinned_allocator_default()


def test_native_allocator_accepts_effective_default(monkeypatch):
    monkeypatch.setenv(KEYS[0], "expandable_segments:False")
    launcher.set_pinned_allocator_default()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; x=torch.ones(1,device='cuda'); assert x.item()==1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
