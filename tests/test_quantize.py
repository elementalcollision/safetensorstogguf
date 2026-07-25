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


if __name__ == "__main__":
    unittest.main()
