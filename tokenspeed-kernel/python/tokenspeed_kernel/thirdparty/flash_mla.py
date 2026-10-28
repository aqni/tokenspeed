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

"""Load the optional FlashMLA extension only when its backend is used."""

from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from types import ModuleType


def is_flash_mla_available() -> bool:
    """Check package presence without importing or validating the CUDA extension."""
    try:
        return importlib.util.find_spec("flash_mla") is not None
    except ModuleNotFoundError as exc:
        if exc.name != "flash_mla":
            raise
        return False


@lru_cache(maxsize=1)
def load_flash_mla() -> ModuleType:
    """Load FlashMLA on demand; preserve native ABI and transitive import errors."""
    try:
        return importlib.import_module("flash_mla")
    except ModuleNotFoundError as exc:
        if exc.name != "flash_mla":
            raise
        raise ImportError(
            "The selected FlashMLA backend requires a compatible flash_mla package"
        ) from exc


def has_flash_mla_v41_api() -> bool:
    """Probe the V4.1 API on demand, after loading the native extension."""
    return is_flash_mla_available() and hasattr(
        load_flash_mla(), "fused_norm_rope_attn_rope_cast"
    )


@lru_cache(maxsize=1)
def load_flash_mla_v41_api() -> ModuleType:
    """Require the V4.1 API without making it a dependency of ordinary DSA."""
    load_flash_mla()
    if not has_flash_mla_v41_api():
        raise ImportError("FlashMLA V4.1 requires fused_norm_rope_attn_rope_cast")
    return importlib.import_module("flash_mla.flash_mla_interface")


def flash_mla_sparse_fwd(*args, **kwargs):
    """Forward sparse prefill arguments unchanged to the optional native API."""
    return load_flash_mla().flash_mla_sparse_fwd(*args, **kwargs)


def flash_mla_with_kvcache(*args, **kwargs):
    """Forward paged attention arguments unchanged to the optional native API."""
    return load_flash_mla().flash_mla_with_kvcache(*args, **kwargs)


def get_mla_metadata(*args, **kwargs):
    """Create native scheduling metadata only for a caller using FlashMLA."""
    return load_flash_mla().get_mla_metadata(*args, **kwargs)
