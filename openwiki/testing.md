---
type: Testing
title: Testing — Regression Suite and Compatibility Tests
description: How to run the test suite, what each test file covers, the regression history behind each test, and guidance for adding new tests when changing the adapter or quantization logic.
tags: [testing, regression, upstream-compat, quantize, unittest]
---

# Testing

## Running the Tests

```bash
python3 -m unittest discover -s tests -v
```

The suite runs in under a second. It needs neither `torch` nor a real llama.cpp checkout. Tests that require the `gguf` Python module skip cleanly when it is unavailable.

## Test Files

### `tests/test_upstream_compat.py`

**Source:** `tests/test_upstream_compat.py`

**Purpose:** Verifies that `safetensors_to_gguf.py`'s `UpstreamConverter` adapter handles all three generations of llama.cpp's converter API.

**How it works:** Builds miniature llama.cpp checkouts in temp directories, each reproducing one generation's public surface:

1. **One-argument monolith** (pre-#14737): `load_hparams(dir_model)`, no `get_model_class`, eager registry.
2. **Two-argument monolith** (#14737–pre-#17114): `load_hparams(dir_model, is_mistral_format)`, no `get_model_class`, eager registry.
3. **`conversion` package** (#17114+): `load_hparams` with `is_mistral_format`, `get_model_class` performing lazy imports, `get_model_architecture` with nested config support.

Each stub checkout includes a minimal `ModelBase` class, a `get_model_architecture` function, a `gguf` module stub with `LlamaFileType`, and registered test model classes. The tests assert that `setup_llama_cpp_path()` + the `UpstreamConverter` methods correctly resolve model classes and load hyperparameters for all three layouts.

**Regression history:**
- **Issue #9** — Post-#17114, model classes moved to a `conversion` package with lazy registration. The old loader hard-failed with "Incompatible llama.cpp" and resolved no architectures.
- **PR #8** — `load_hparams` gained a required `is_mistral_format` argument in #14737 (2025-08-11). The adapter must call it with the correct signature.

### `tests/test_quantize.py`

**Source:** `tests/test_quantize.py`

**Purpose:** Regression tests for `quantize_gguf.py`'s analysis and MoE tensor-targeting logic.

**How it works:** Uses `gguf.GGUFWriter` to create tiny but structurally valid GGUF files with named tensors (dense and MoE layouts). Tests exercise:

- `moe_tensor_type_args()` — Verifies that `--moe-expert-quantization` / `--moe-router-quantization` produce correct `--tensor-type NAME=TYPE` arguments.
  - `same` emits nothing.
  - Expert type covers all three expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`).
  - Router type targets `ffn_gate_inp`.
  - Both combine to produce 4 well-formed `--tensor-type` pairs.
- `analyze_model_structure()` — Verifies correct MoE detection:
  - Dense FFN tensors (`ffn_gate`, `ffn_up`, `ffn_down` without `_exps`) must NOT be detected as MoE.
  - Tensors with `_exps` suffix and `ffn_gate_inp` are correctly identified as MoE expert and router tensors.

**Regression history:**
- The previous `analyze_model_structure()` shelled out to `llama-quantize --dry-run --verbose` and regex-scraped stdout. This never worked: `--verbose` is not a `llama-quantize` flag (exits 1), output goes to stderr, and the format didn't match the regex. The function always returned `{"error": ...}`.
- The `--moe-*` flags were declared but documented as ignored. Upstream `llama-quantize` now supports `--tensor-type NAME=TYPE`, so the tests verify they map correctly.

## Guidance for Adding Tests

### When changing the UpstreamConverter adapter

Add a new stub checkout variant in `test_upstream_compat.py` if llama.cpp introduces a fourth converter layout. Follow the pattern: write a `_BASE_PY` template, create the directory structure with `_write()`, and assert `setup_llama_cpp_path()` + `UpstreamConverter` methods work against it.

### When changing quantization logic

Add test cases in `test_quantize.py` using the `write_gguf()` helper to create fixtures. The helper writes minimal GGUF files with `GGUFWriter` — no model weights, torch, or compiled binary needed. Test the function directly (e.g., `moe_tensor_type_args()`) rather than shelling out to `llama-quantize`.

### When adding new CLI flags

Add argument parsing tests that verify the flag is accepted and routed to the correct `llama-quantize` argument. If the flag affects analysis, add a test fixture with the relevant tensor names.

## Relationship to Workflows

The tests guard the two most regression-prone areas of the [workflows](workflows.md): the adapter that interfaces with llama.cpp's evolving API (exercised by the conversion workflow), and the MoE analysis + per-tensor targeting logic (exercised by the quantization workflow). When modifying either workflow, run the test suite first to establish a baseline.
