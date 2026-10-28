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

"""CPU checks for optional FlashMLA loading, independent of Torch/Triton.

Native GPU packages are represented by temporary modules in isolated Python
processes. The real adapter and MLA facade run without importing the rest of
the GPU stack. Run directly with Python when pytest's GPU conftest is unavailable.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_KERNEL = Path(__file__).resolve().parents[1] / "python" / "tokenspeed_kernel"
_ADAPTER = _KERNEL / "thirdparty" / "flash_mla.py"
_FACADE = _KERNEL / "ops" / "attention" / "mla" / "cuda.py"
_BOOTSTRAP = """
import importlib.util
import sys

sys.path.insert(0, sys.argv[3])

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

adapter = load("tokenspeed_kernel.thirdparty.flash_mla", sys.argv[1])
facade = load("flashmla_facade_under_test", sys.argv[2])
"""
_NATIVE = """
def flash_mla_sparse_fwd(*args, **kwargs):
    return args, kwargs

def flash_mla_with_kvcache(*args, **kwargs):
    return args, kwargs

def get_mla_metadata(*args, **kwargs):
    return args, kwargs
"""


class FlashMLAOptionalTests(unittest.TestCase):
    def probe(self, native_source, script, *, interface=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if native_source is not None:
                package = root / "flash_mla"
                package.mkdir()
                (package / "__init__.py").write_text(textwrap.dedent(native_source))
                if interface is not None:
                    (package / "flash_mla_interface.py").write_text(
                        textwrap.dedent(interface)
                    )
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _BOOTSTRAP + textwrap.dedent(script),
                    str(_ADAPTER),
                    str(_FACADE),
                    directory,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_absent_package_does_not_block_facade_import(self):
        self.probe(
            None,
            """
            assert not adapter.is_flash_mla_available()
            assert not adapter.has_flash_mla_v41_api()
            for call in (
                facade.get_mla_metadata,
                facade.flash_mla_sparse_fwd,
                facade.flash_mla_with_kvcache,
                adapter.load_flash_mla_v41_api,
            ):
                try:
                    call()
                except ImportError as exc:
                    assert "requires a compatible flash_mla package" in str(exc)
                else:
                    raise AssertionError("missing native package was accepted")
            """,
        )

    def test_native_abi_failure_is_deferred_and_preserved(self):
        for exception in ("ImportError", "OSError"):
            with self.subTest(exception=exception):
                self.probe(
                    f'raise {exception}("native ABI mismatch")',
                    f"""
                    assert adapter.is_flash_mla_available()
                    try:
                        facade.get_mla_metadata()
                    except {exception} as exc:
                        assert str(exc) == "native ABI mismatch"
                    else:
                        raise AssertionError("native failure was swallowed")
                    """,
                )

    def test_transitive_missing_dependency_is_preserved(self):
        self.probe(
            'raise ModuleNotFoundError("missing CUDA dependency", name="cuda_dependency")',
            """
            assert adapter.is_flash_mla_available()
            try:
                facade.get_mla_metadata()
            except ModuleNotFoundError as exc:
                assert exc.name == "cuda_dependency"
                assert str(exc) == "missing CUDA dependency"
            else:
                raise AssertionError("transitive import failure was swallowed")
            """,
        )

    def test_arguments_and_native_results_are_forwarded(self):
        self.probe(
            _NATIVE,
            """
            for call in (
                facade.get_mla_metadata,
                facade.flash_mla_sparse_fwd,
                facade.flash_mla_with_kvcache,
            ):
                query, schedule, output = object(), object(), object()
                args, kwargs = call(
                    query, tile_scheduler_metadata=schedule, out=output, causal=False
                )
                assert args == (query,)
                assert kwargs == {
                    "tile_scheduler_metadata": schedule, "out": output, "causal": False
                }
            """,
        )

    def test_missing_v41_api_does_not_disable_dsa(self):
        self.probe(
            _NATIVE,
            """
            assert not adapter.has_flash_mla_v41_api()
            try:
                adapter.load_flash_mla_v41_api()
            except ImportError as exc:
                assert "fused_norm_rope_attn_rope_cast" in str(exc)
            else:
                raise AssertionError("missing V4.1 API was accepted")
            assert facade.flash_mla_sparse_fwd(123) == ((123,), {})
            """,
        )

    def test_v41_interface_is_loaded_when_requested(self):
        self.probe(
            _NATIVE + "\nfused_norm_rope_attn_rope_cast = object()\n",
            """
            assert adapter.has_flash_mla_v41_api()
            api = adapter.load_flash_mla_v41_api()
            assert api.get_mla_metadata() == "v41 schedule"
            """,
            interface="""
            def get_mla_metadata():
                return "v41 schedule"
            """,
        )

    def test_v41_interface_error_does_not_block_dsa(self):
        self.probe(
            _NATIVE + "\nfused_norm_rope_attn_rope_cast = object()\n",
            """
            assert facade.flash_mla_sparse_fwd(123) == ((123,), {})
            try:
                adapter.load_flash_mla_v41_api()
            except ImportError as exc:
                assert str(exc) == "v41 extension unavailable"
            else:
                raise AssertionError("V4.1 import failure was swallowed")
            """,
            interface='raise ImportError("v41 extension unavailable")',
        )


if __name__ == "__main__":
    unittest.main()
