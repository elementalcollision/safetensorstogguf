#!/usr/bin/env python3
"""Tests for quantize_gguf.py.

Fixtures are tiny GGUF files written with gguf.GGUFWriter, so these need no
model weights, no torch and no compiled llama-quantize binary. They skip
cleanly when the gguf module is unavailable.

Regression coverage:
  * analyze_model_structure() used to shell out to
    `llama-quantize --dry-run --verbose ... /dev/null q4_0` and regex-scrape the
    result. `--verbose` is not a llama-quantize flag (it exits 1), llama.cpp logs
    to stderr rather than stdout, and the dry-run output carries per-type totals
    rather than the per-tensor lines the pattern expected - so the function could
    never return anything but {"error": ...}.
  * --moe-expert-quantization / --moe-router-quantization were declared but
    documented as ignored. Upstream now has --tensor-type NAME=TYPE, so they map
    onto it directly.

Run with:  python3 -m unittest discover -s tests -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import quantize_gguf  # noqa: E402

try:
    quantize_gguf._add_gguf_to_path()
    import gguf  # noqa: F401
    HAVE_GGUF = True
except Exception:  # pragma: no cover - environment without gguf
    HAVE_GGUF = False


def write_gguf(path: Path, tensor_names, arch="llama", cols=32, rows=2):
    """Write a minimal but structurally valid GGUF containing named tensors."""
    import numpy as np
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), arch)
    writer.add_block_count(1)
    for name in tensor_names:
        writer.add_tensor(name, np.zeros((rows, cols), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


DENSE_TENSORS = [
    "token_embd.weight",
    "blk.0.attn_q.weight",
    "blk.0.ffn_gate.weight",   # dense FFN - must NOT read as MoE
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
    "output.weight",
]

MOE_TENSORS = DENSE_TENSORS + [
    "blk.0.ffn_gate_exps.weight",
    "blk.0.ffn_up_exps.weight",
    "blk.0.ffn_down_exps.weight",
    "blk.0.ffn_gate_inp.weight",   # router
]


class TestMoeTensorTypeArgs(unittest.TestCase):
    """--moe-* flags translate into upstream --tensor-type arguments."""

    def test_same_emits_nothing(self):
        self.assertEqual(quantize_gguf.moe_tensor_type_args("same", "same"), [])

    def test_expert_type_covers_all_three_expert_tensors(self):
        args = quantize_gguf.moe_tensor_type_args("q4_k", "same")
        self.assertEqual(args.count("--tensor-type"), 3)
        for name in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            self.assertIn(f"{name}=q4_k", args)

    def test_router_type_targets_the_router_tensor(self):
        args = quantize_gguf.moe_tensor_type_args("same", "f32")
        self.assertEqual(args, ["--tensor-type", "ffn_gate_inp=f32"])

    def test_both_combine(self):
        args = quantize_gguf.moe_tensor_type_args("q4_k", "f32")
        self.assertEqual(args.count("--tensor-type"), 4)
        self.assertIn("ffn_gate_inp=f32", args)

    def test_pairs_are_well_formed(self):
        args = quantize_gguf.moe_tensor_type_args("q8_0", "f16")
        # Every flag is followed by exactly one NAME=TYPE value.
        for i in range(0, len(args), 2):
            self.assertEqual(args[i], "--tensor-type")
            self.assertRegex(args[i + 1], r"^[a-z_]+=[a-z0-9_]+$")


@unittest.skipUnless(HAVE_GGUF, "gguf module not available")
class TestReadGgufTensors(unittest.TestCase):
    """Tensor metadata is read from the file, not scraped from a subprocess."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_every_tensor(self):
        path = write_gguf(self.tmp / "m.gguf", DENSE_TENSORS)
        tensors = quantize_gguf.read_gguf_tensors(path)
        self.assertEqual(len(tensors), len(DENSE_TENSORS))
        self.assertEqual({t["name"] for t in tensors}, set(DENSE_TENSORS))

    def test_shape_dtype_and_size_are_populated(self):
        path = write_gguf(self.tmp / "m.gguf", ["output.weight"], cols=64, rows=4)
        tensor = quantize_gguf.read_gguf_tensors(path)[0]
        self.assertEqual(len(tensor["dimensions"]), 4)      # padded to 4
        self.assertIn(64, tensor["dimensions"])
        self.assertEqual(tensor["type"], "f32")
        self.assertGreater(tensor["size_mb"], 0)

    def test_needs_no_subprocess(self):
        # Regression: the old implementation required the llama-quantize binary
        # and returned {"error": "llama-quantize binary not found"} without it.
        path = write_gguf(self.tmp / "m.gguf", DENSE_TENSORS)
        result = quantize_gguf.analyze_model_structure(path)
        self.assertNotIn("error", result)
        self.assertEqual(result["tensor_count"], len(DENSE_TENSORS))


@unittest.skipUnless(HAVE_GGUF, "gguf module not available")
class TestMoeDetection(unittest.TestCase):
    """MoE detection keys on real GGUF tensor names."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dense_model_is_not_moe(self):
        # Regression: the old keyword list included 'gate', 'ffn_up' and
        # 'ffn_down', so every dense feed-forward model reported has_moe=True
        # and silently got MoE-specific quantization settings applied.
        path = write_gguf(self.tmp / "dense.gguf", DENSE_TENSORS)
        result = quantize_gguf.analyze_model_structure(path)
        self.assertFalse(result["has_moe"])
        self.assertEqual(result["expert_tensors"], [])

    def test_moe_model_is_detected(self):
        path = write_gguf(self.tmp / "moe.gguf", MOE_TENSORS)
        result = quantize_gguf.analyze_model_structure(path)
        self.assertTrue(result["has_moe"])
        names = {t["name"] for t in result["expert_tensors"]}
        self.assertEqual(names, {
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        })

    def test_router_is_classified_separately(self):
        path = write_gguf(self.tmp / "moe.gguf", MOE_TENSORS)
        result = quantize_gguf.analyze_model_structure(path)
        self.assertIn("blk.0.ffn_gate_inp.weight",
                      {t["name"] for t in result["router_tensors"]})
        # ...and is not double-counted as an expert.
        self.assertNotIn("blk.0.ffn_gate_inp.weight",
                         {t["name"] for t in result["expert_tensors"]})

    def test_router_alone_is_not_an_moe(self):
        path = write_gguf(self.tmp / "r.gguf",
                          DENSE_TENSORS + ["blk.0.ffn_gate_inp.weight"])
        self.assertFalse(quantize_gguf.analyze_model_structure(path)["has_moe"])


@unittest.skipUnless(HAVE_GGUF, "gguf module not available")
class TestGgufMetadata(unittest.TestCase):
    """Expert counts come from GGUF metadata, not from tensor names."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_moe(self, path, experts=8, used=2):
        import numpy as np
        from gguf import GGUFWriter
        w = GGUFWriter(str(path), "llama")
        w.add_block_count(1)
        w.add_expert_count(experts)
        w.add_expert_used_count(used)
        for name in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_gate_inp.weight"):
            w.add_tensor(name, np.zeros((2, 32), dtype=np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        return path

    def test_reads_architecture(self):
        path = write_gguf(self.tmp / "m.gguf", DENSE_TENSORS)
        self.assertEqual(quantize_gguf.read_gguf_metadata(path)["architecture"], "llama")

    def test_reads_expert_counts(self):
        # Experts are stacked into one tensor per projection, so the count is
        # only available from metadata - it cannot be derived from names.
        path = self._write_moe(self.tmp / "moe.gguf", experts=256, used=8)
        meta = quantize_gguf.read_gguf_metadata(path)
        self.assertEqual(meta["expert_count"], 256)
        self.assertEqual(meta["expert_used_count"], 8)

    def test_dense_model_has_no_expert_count(self):
        path = write_gguf(self.tmp / "d.gguf", DENSE_TENSORS)
        self.assertIn(quantize_gguf.read_gguf_metadata(path).get("expert_count"), (None, 0))


class TestConvertAndQuantizeSharesMoeMapping(unittest.TestCase):
    """convert_and_quantize.py must not carry its own copy of the mapping."""

    def test_imports_the_shared_helper(self):
        import convert_and_quantize
        self.assertIs(convert_and_quantize.moe_tensor_type_args,
                      quantize_gguf.moe_tensor_type_args)

    def test_help_no_longer_claims_the_flags_are_ignored(self):
        source = (REPO_ROOT / "convert_and_quantize.py").read_text()
        self.assertNotIn("currently ignored", source)
        self.assertNotIn("not supported by upstream llama-quantize", source)


class TestImatrixArgs(unittest.TestCase):
    """--imatrix and its per-tensor scoping map onto llama-quantize flags."""

    class _Args:
        def __init__(self, imatrix=None, include=None, exclude=None, type="q4_k"):
            self.imatrix = imatrix
            self.include_weights = include
            self.exclude_weights = exclude
            self.type = type

    def test_no_imatrix_emits_nothing(self):
        self.assertEqual(quantize_gguf.imatrix_args(self._Args()), [])

    def test_imatrix_is_forwarded(self):
        args = self._Args(imatrix="/tmp/m.imatrix")
        self.assertEqual(quantize_gguf.imatrix_args(args),
                         ["--imatrix", "/tmp/m.imatrix"])

    def test_include_weights_is_repeatable(self):
        args = self._Args(imatrix="/m", include=["attn_q", "ffn_up"])
        self.assertEqual(quantize_gguf.imatrix_args(args),
                         ["--imatrix", "/m",
                          "--include-weights", "attn_q",
                          "--include-weights", "ffn_up"])

    def test_exclude_weights_is_repeatable(self):
        args = self._Args(imatrix="/m", exclude=["attn_q", "attn_k"])
        self.assertEqual(quantize_gguf.imatrix_args(args),
                         ["--imatrix", "/m",
                          "--exclude-weights", "attn_q",
                          "--exclude-weights", "attn_k"])

    def test_scoping_without_imatrix_emits_nothing(self):
        # validate_imatrix_args rejects this combination up front; if it ever
        # slipped through, do not emit dangling scope flags.
        args = self._Args(include=["attn_q"])
        self.assertEqual(quantize_gguf.imatrix_args(args), [])


class TestImatrixValidation(unittest.TestCase):
    """Impossible combinations are rejected before any work happens."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.imatrix = self.tmp / "m.imatrix"
        self.imatrix.write_bytes(b"\x00")
        self.errors = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fail(self, message):
        self.errors.append(message)

    def _check(self, **kw):
        args = TestImatrixArgs._Args(**kw)
        quantize_gguf.validate_imatrix_args(self._fail, args)
        return self.errors

    def test_accepts_a_valid_combination(self):
        self.assertEqual(self._check(imatrix=self.imatrix, include=["attn_q"]), [])

    def test_rejects_include_with_exclude(self):
        errs = self._check(imatrix=self.imatrix, include=["a"], exclude=["b"])
        self.assertTrue(any("cannot be used together" in e for e in errs))

    def test_rejects_scoping_without_imatrix(self):
        errs = self._check(include=["attn_q"])
        self.assertTrue(any("require --imatrix" in e for e in errs))

    def test_rejects_missing_imatrix_file(self):
        errs = self._check(imatrix=self.tmp / "nope.imatrix")
        self.assertTrue(any("not found" in e for e in errs))

    def test_rejects_low_bit_type_without_imatrix(self):
        # llama.cpp raises "this quantization requires an imatrix!" only after
        # loading the model; fail before that work is done.
        for quant in sorted(quantize_gguf.IMATRIX_REQUIRED_TYPES):
            self.errors = []
            errs = self._check(type=quant)
            self.assertTrue(any("requires an importance matrix" in e for e in errs),
                            f"{quant} should require an imatrix")

    def test_low_bit_type_with_imatrix_is_accepted(self):
        self.assertEqual(self._check(imatrix=self.imatrix, type="iq2_xxs"), [])

    def test_ordinary_types_need_no_imatrix(self):
        for quant in ("q4_k", "q8_0", "iq4_nl", "iq4_xs", "iq3_s", "q2_k"):
            self.errors = []
            self.assertEqual(self._check(type=quant), [], f"{quant} must not require one")


class TestConvertAndQuantizeSharesImatrix(unittest.TestCase):
    """Both drivers use one implementation, not two copies."""

    def test_imports_the_shared_helpers(self):
        import convert_and_quantize
        self.assertIs(convert_and_quantize.imatrix_args, quantize_gguf.imatrix_args)
        self.assertIs(convert_and_quantize.validate_imatrix_args,
                      quantize_gguf.validate_imatrix_args)


class TestTypeAuto(unittest.TestCase):
    """`auto` selects analysis-only mode; it is never a quantization type."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "m.gguf"
        self.model.write_bytes(b"GGUF")   # never opened: auto short-circuits first

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *extra):
        import subprocess
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "quantize_gguf.py"),
             "--model", str(self.model), *extra],
            capture_output=True, text=True)

    def test_auto_alone_enters_analysis_mode(self):
        # Regression: this used to fall through and forward "auto" to
        # llama-quantize, which rejects it with "invalid ftype 'auto'".
        out = self._run("--type", "auto")
        combined = out.stdout + out.stderr
        self.assertIn("analysis-only", combined)
        self.assertNotIn("invalid ftype", combined)

    def test_auto_never_invokes_the_quantizer(self):
        out = self._run("--type", "auto")
        self.assertNotIn("Running quantization command", out.stdout + out.stderr)

    def test_auto_with_analyze_model_is_unchanged(self):
        out = self._run("--type", "auto", "--analyze-model")
        self.assertIn("analysis-only", out.stdout + out.stderr)

    def test_auto_is_still_an_accepted_choice(self):
        # argparse must not reject it; it is a documented mode selector.
        self.assertIn("auto", self._run("--help").stdout)


if __name__ == "__main__":
    unittest.main()
