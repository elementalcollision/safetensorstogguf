#!/usr/bin/env python3
"""Compatibility tests for the llama.cpp converter adapter.

These build miniature llama.cpp checkouts reproducing each generation of the
converter's public surface and assert that ``safetensors_to_gguf`` adapts to all
of them. They need neither torch nor a real llama.cpp.

Regression coverage:
  * issue #9  — post-llama.cpp#17114 the model classes moved into a `conversion`
                package and register lazily; the old loader hard-failed with
                "Incompatible llama.cpp" and resolved no architectures.
  * PR #8     — ``load_hparams`` gained a required ``is_mistral_format``
                argument in llama.cpp#14737 (2025-08-11).

Run with:  python3 -m unittest discover -s tests -v
"""

import importlib
import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Stub llama.cpp checkouts
# --------------------------------------------------------------------------

_BASE_PY = '''
import json
from enum import IntEnum
from pathlib import Path


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


class ModelBase:
    _model_classes = {ModelType.TEXT: {}, ModelType.MMPROJ: {}}
    is_mistral_format = False

    @staticmethod
    def load_hparams(dir_model, is_mistral_format):
        name = "params.json" if is_mistral_format else "config.json"
        with open(Path(dir_model) / name, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def register(cls, *names, model_type=ModelType.TEXT):
        def decorator(klass):
            for name in names:
                cls._model_classes[model_type][name] = klass
            return klass
        return decorator

    @classmethod
    def from_model_architecture(cls, arch, model_type=ModelType.TEXT):
        try:
            return cls._model_classes[model_type][arch]
        except KeyError:
            raise NotImplementedError(f"Architecture {arch!r} not supported!") from None


def get_model_architecture(hparams, model_type=ModelType.TEXT):
    text_config = hparams.get("text_config", {})
    arch = None
    architectures = hparams.get("architectures")
    if architectures:
        arch = architectures[0]
    if text_config.get("architectures"):
        arch = text_config["architectures"][0]
    if arch is None:
        raise ValueError("could not determine architecture")
    return arch
'''


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


def make_modern_checkout(root: Path) -> Path:
    """llama.cpp at/after #17114: `conversion` package, lazy registry, no model
    classes re-exported from convert_hf_to_gguf.py."""
    _write(root / "gguf-py" / "gguf.py", "class LlamaFileType:\n    MOSTLY_F16 = 1\n")
    _write(root / "conversion" / "base.py", _BASE_PY)
    _write(root / "conversion" / "__init__.py", '''
        from .base import ModelBase, ModelType, get_model_architecture

        TEXT_MODEL_MAP = {
            "LlamaForCausalLM": "llama",
            "DeepseekV4ForCausalLM": "deepseek",
        }
        MMPROJ_MODEL_MAP = {"PixtralVisionModel": "pixtral"}

        _mistral_common_installed = True
        _mistral_import_error_msg = "pip install mistral-common"


        def get_model_class(name, mmproj=False):
            mapping = MMPROJ_MODEL_MAP if mmproj else TEXT_MODEL_MAP
            if name not in mapping:
                raise NotImplementedError(f"Architecture {name!r} not supported!")
            __import__(f"conversion.{mapping[name]}")
            model_type = ModelType.MMPROJ if mmproj else ModelType.TEXT
            return ModelBase._model_classes[model_type][name]
    ''')
    _write(root / "conversion" / "llama.py", '''
        from .base import ModelBase


        @ModelBase.register("LlamaForCausalLM")
        class LlamaModel(ModelBase):
            pass
    ''')
    _write(root / "conversion" / "deepseek.py", '''
        from .base import ModelBase


        @ModelBase.register("DeepseekV4ForCausalLM")
        class DeepseekV4Model(ModelBase):
            pass
    ''')
    _write(root / "conversion" / "mistral.py", '''
        from .base import ModelBase


        class MistralModel(ModelBase):
            is_mistral_format = True


        class MistralMoeModel(ModelBase):
            is_mistral_format = True
    ''')
    _write(root / "conversion" / "pixtral.py", '''
        from .base import ModelBase, ModelType


        @ModelBase.register("PixtralVisionModel", model_type=ModelType.MMPROJ)
        class PixtralModel(ModelBase):
            is_mistral_format = True
    ''')
    # The real post-#17114 entrypoint re-exports helpers only - no model classes.
    _write(root / "convert_hf_to_gguf.py", '''
        from conversion import (
            ModelBase,
            ModelType,
            get_model_architecture,
            get_model_class,
            _mistral_common_installed,
            _mistral_import_error_msg,
        )
    ''')
    return root


def make_monolithic_checkout(root: Path, *, mistral_arg: bool = True) -> Path:
    """Pre-#17114 monolith. Everything in one module, registry populated eagerly.

    With ``mistral_arg=False`` this reproduces llama.cpp before #14737
    (2025-08-11), when ``load_hparams`` took a single argument.
    """
    _write(root / "gguf-py" / "gguf.py", "class LlamaFileType:\n    MOSTLY_F16 = 1\n")
    body = _BASE_PY
    if not mistral_arg:
        body = body.replace(
            "    def load_hparams(dir_model, is_mistral_format):\n"
            '        name = "params.json" if is_mistral_format else "config.json"\n',
            "    def load_hparams(dir_model):\n"
            '        name = "config.json"\n',
        )
    extra = '''

@ModelBase.register("LlamaForCausalLM")
class LlamaModel(ModelBase):
    pass


@ModelBase.register("DeepseekV4ForCausalLM")
class DeepseekV4Model(ModelBase):
    pass


class MistralModel(ModelBase):
    is_mistral_format = True


class MistralMoeModel(ModelBase):
    is_mistral_format = True
'''
    _write(root / "convert_hf_to_gguf.py", body + extra)
    return root


def make_model_dir(root: Path, *, mistral: bool = False, moe: bool = False,
                   arch: str = "LlamaForCausalLM", nested: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model-00001-of-00001.safetensors").write_bytes(b"\x00")
    if mistral:
        params = {"dim": 4096, "n_layers": 32, "n_heads": 32, "vocab_size": 32000}
        if moe:
            params["moe"] = {"num_experts": 8}
        (root / "params.json").write_text(json.dumps(params), encoding="utf-8")
    else:
        config = ({"text_config": {"architectures": [arch]}} if nested
                  else {"architectures": [arch]})
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


# --------------------------------------------------------------------------


class UpstreamCompatTestCase(unittest.TestCase):
    """Base class handling the module/sys.path juggling the stubs require."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._sys_path = list(sys.path)
        self._sys_modules = set(sys.modules)
        self.stg = importlib.import_module("safetensors_to_gguf")

    def tearDown(self):
        sys.path[:] = self._sys_path
        for name in set(sys.modules) - self._sys_modules:
            del sys.modules[name]
        self.stg.LLAMA_CPP_PATH = None
        self.stg.Model = None
        self.stg.UPSTREAM = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self, checkout: Path):
        self.stg.setup_llama_cpp_path(checkout)
        return self.stg.UPSTREAM


class TestModernCheckout(UpstreamCompatTestCase):
    """llama.cpp at/after #17114 - the layout that broke this tool (issue #9)."""

    def setUp(self):
        super().setUp()
        self.checkout = make_modern_checkout(self.tmp / "llama.cpp")

    def test_loader_accepts_checkout_without_model_classes(self):
        # Previously raised ImportError("Incompatible llama.cpp ... no LlamaModel").
        upstream = self.load(self.checkout)
        self.assertEqual(upstream.layout, "conversion-package")
        self.assertTrue(upstream.has_lazy_registry)

    def test_resolves_architecture_needing_lazy_import(self):
        # issue #9: DeepseekV4 lives in conversion/deepseek.py and is registered
        # only once that module is imported. from_model_architecture cannot see it.
        upstream = self.load(self.checkout)
        with self.assertRaises(NotImplementedError):
            upstream.model_base.from_model_architecture("DeepseekV4ForCausalLM")
        self.assertEqual(
            upstream.model_class("DeepseekV4ForCausalLM").__name__, "DeepseekV4Model"
        )

    def test_resolves_plain_llama(self):
        upstream = self.load(self.checkout)
        self.assertEqual(upstream.model_class("LlamaForCausalLM").__name__, "LlamaModel")

    def test_unsupported_architecture_still_raises(self):
        upstream = self.load(self.checkout)
        with self.assertRaises(NotImplementedError):
            upstream.model_class("NotARealArch")

    def test_architecture_resolved_via_upstream_resolver(self):
        upstream = self.load(self.checkout)
        # Nested under text_config: a bare hparams["architectures"][0] misses this.
        hparams = {"text_config": {"architectures": ["DeepseekV4ForCausalLM"]}}
        self.assertEqual(upstream.model_architecture(hparams), "DeepseekV4ForCausalLM")

    def test_load_hparams_passes_mistral_flag(self):
        upstream = self.load(self.checkout)
        model_dir = make_model_dir(self.tmp / "mistral", mistral=True)
        hparams = upstream.load_hparams(model_dir, True)
        self.assertEqual(hparams["dim"], 4096)
        self.assertNotIn("architectures", hparams)

    def test_mistral_class_selection(self):
        upstream = self.load(self.checkout)
        self.assertEqual(upstream.mistral_model_class({}).__name__, "MistralModel")
        self.assertEqual(
            upstream.mistral_model_class({"moe": {"num_experts": 8}}).__name__,
            "MistralMoeModel",
        )

    def test_mistral_mmproj_requires_vision_encoder(self):
        upstream = self.load(self.checkout)
        with self.assertRaises(ValueError):
            upstream.mistral_model_class({}, mmproj=True)
        self.assertEqual(
            upstream.mistral_model_class({"vision_encoder": {}}, mmproj=True).__name__,
            "PixtralModel",
        )


class TestMonolithicCheckout(UpstreamCompatTestCase):
    """Pre-#17114 llama.cpp must keep working."""

    def setUp(self):
        super().setUp()
        self.checkout = make_monolithic_checkout(self.tmp / "llama.cpp")

    def test_loader_uses_monolith(self):
        upstream = self.load(self.checkout)
        self.assertEqual(upstream.layout, "monolithic")
        self.assertFalse(upstream.has_lazy_registry)

    def test_resolves_via_eager_registry(self):
        upstream = self.load(self.checkout)
        self.assertEqual(upstream.model_class("LlamaForCausalLM").__name__, "LlamaModel")
        self.assertEqual(
            upstream.model_class("DeepseekV4ForCausalLM").__name__, "DeepseekV4Model"
        )

    def test_mistral_classes_come_from_monolith(self):
        upstream = self.load(self.checkout)
        self.assertEqual(upstream.mistral_model_class({}).__name__, "MistralModel")


class TestLegacySignature(UpstreamCompatTestCase):
    """llama.cpp before #14737 (2025-08-11): load_hparams took one argument."""

    def setUp(self):
        super().setUp()
        self.checkout = make_monolithic_checkout(
            self.tmp / "llama.cpp", mistral_arg=False
        )

    def test_load_hparams_omits_unsupported_argument(self):
        upstream = self.load(self.checkout)
        model_dir = make_model_dir(self.tmp / "hf")
        hparams = upstream.load_hparams(model_dir, False)
        self.assertEqual(hparams["architectures"], ["LlamaForCausalLM"])

    def test_mistral_request_fails_with_actionable_error(self):
        upstream = self.load(self.checkout)
        model_dir = make_model_dir(self.tmp / "mistral", mistral=True)
        with self.assertRaises(ValueError) as ctx:
            upstream.load_hparams(model_dir, True)
        self.assertIn("14737", str(ctx.exception))


class TestMistralDetection(unittest.TestCase):
    """--mistral-format resolution is pure argument/layout logic."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        sys.path.insert(0, str(REPO_ROOT))
        self.stg = importlib.import_module("safetensors_to_gguf")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    class _Args:
        def __init__(self, model, mistral_format=None):
            self.model = model
            self.mistral_format = mistral_format

    def test_autodetects_mistral_layout(self):
        model_dir = make_model_dir(self.tmp / "mistral", mistral=True)
        self.assertTrue(self.stg.resolve_mistral_format(self._Args(model_dir)))

    def test_hf_layout_is_not_mistral(self):
        model_dir = make_model_dir(self.tmp / "hf")
        self.assertFalse(self.stg.resolve_mistral_format(self._Args(model_dir)))

    def test_both_files_present_prefers_huggingface(self):
        model_dir = make_model_dir(self.tmp / "both")
        (model_dir / "params.json").write_text("{}", encoding="utf-8")
        self.assertFalse(self.stg.resolve_mistral_format(self._Args(model_dir)))

    def test_explicit_flags_override_detection(self):
        hf_dir = make_model_dir(self.tmp / "hf2")
        mistral_dir = make_model_dir(self.tmp / "mistral2", mistral=True)
        self.assertTrue(self.stg.resolve_mistral_format(self._Args(hf_dir, True)))
        self.assertFalse(self.stg.resolve_mistral_format(self._Args(mistral_dir, False)))


if __name__ == "__main__":
    unittest.main()
