---
type: Architecture
title: Architecture — Adapter Pattern and llama.cpp Compatibility
description: How the toolkit probes and adapts to multiple generations of llama.cpp's converter API, the UpstreamConverter class, module dependency graph, and the version compatibility matrix.
tags: [architecture, adapter-pattern, upstream-converter, llama-cpp, compatibility]
---

# Architecture

The toolkit is a set of standalone CLI scripts with no shared Python package. Each script is self-contained; the only cross-script coupling is `convert_and_quantize.py` calling `safetensors_to_gguf.py` as a subprocess. The architectural centerpiece is the **UpstreamConverter** adapter in `safetensors_to_gguf.py`, which probes the llama.cpp checkout at runtime and adapts to its converter API shape.

## UpstreamConverter — The Adapter

**Source:** `safetensors_to_gguf.py`, class `UpstreamConverter` (line ~32)

llama.cpp's `convert_hf_to_gguf.py` has been refactored several times. Rather than pinning one API shape, `UpstreamConverter` detects which capabilities the checkout provides and delegates accordingly.

### Three generations supported

| Generation | llama.cpp version | `load_hparams` signature | Class registry | Status |
|---|---|---|---|---|
| Pre-#14737 | before `a3a7874` (2025-08-11) | `load_hparams(dir_model)` | Eager (monolithic module) | Supported (no Mistral format) |
| #14737 → pre-#17114 | `a3a7874` → `cc7200bf` | `load_hparams(dir_model, is_mistral_format)` | Eager (monolithic module) | Supported |
| #17114+ | `cc7200bf` (2026-05-15)+ | Two-arg `load_hparams` | Lazy (`conversion` package, `get_model_class`) | Supported |

### Key methods

- **`load_hparams(dir_model, is_mistral_format)`** — Inspects the signature of `ModelBase.load_hparams` and calls it with or without the `is_mistral_format` argument. Raises `ValueError` if Mistral format is requested on a pre-#14737 checkout.
- **`model_architecture(hparams, mmproj)`** — Delegates to upstream's `get_model_architecture`, which understands nested config layouts (`text_config`, `llm_config`, `thinker_config`, `ssm_cfg`). Falls back to `hparams["architectures"][0]` on very old checkouts.
- **`model_class(architecture, mmproj)`** — Uses `get_model_class` (lazy import) when available, or `from_model_architecture` (eager registry) on legacy monolithic modules. This is what makes all 234 llama.cpp architectures available.
- **`mistral_model_class(hparams, mmproj)`** — Selects `PixtralModel` (mmproj), `MistralMoeModel` (moe key present), or `MistralModel` from the `conversion.mistral` / `conversion.pixtral` submodule.

### Capability probing flow

1. `setup_llama_cpp_path()` locates the llama.cpp directory (auto-detect or `--llama-cpp-dir`).
2. It tries importing the `conversion` package (post-#17114). If found, it verifies the package file is inside the llama.cpp checkout to avoid colliding with an unrelated `conversion` package.
3. If no `conversion` package, it loads the monolithic `convert_hf_to_gguf.py` by file path (avoiding `__main__` execution).
4. Resolves `ModelBase` (or legacy `Model`) as the base class.
5. Constructs `UpstreamConverter(Model, module, package)` with whichever sources were found.

**Source:** `safetensors_to_gguf.py`, `setup_llama_cpp_path()` (line ~185)

## File Dependency Graph

```
safetensors_to_gguf.py
  ├── imports: gguf (from llama.cpp), llama.cpp convert_hf_to_gguf / conversion package
  ├── imports: torch (at conversion time), transformers/safetensors/sentencepiece (transitively via llama.cpp)
  └── called by: convert_and_quantize.py (as subprocess)

quantize_gguf.py
  ├── imports: gguf (for read_gguf_tensors / analysis), numpy
  ├── subprocess: llama-quantize binary
  └── standalone (not imported by other scripts)

convert_and_quantize.py
  ├── subprocess: python safetensors_to_gguf.py (conversion step)
  ├── subprocess: llama-quantize binary (quantization step)
  └── standalone

analyze_gguf.py        → imports gguf, numpy
analyze_gguf_simple.py → subprocess llama-quantize
analyze_model.py       → subprocess llama-quantize --dry-run
```

No script imports another as a Python module. Cross-script interaction is always via subprocess.

## Module Discovery Pattern

All scripts that need llama.cpp share the same discovery logic, replicated per-script (not shared):

1. `--llama-cpp-dir` CLI argument (explicit).
2. `LLAMA_CPP_DIR` environment variable.
3. Relative to the script: `script_dir.parent.parent`, `script_dir.parent`, `script_dir`.
4. For the `gguf` Python module: checks for `gguf-py/gguf/` directory in candidate paths.
5. For the `llama-quantize` binary: checks `llama-quantize`, `build/llama-quantize`, and `build/bin/llama-quantize` (with `.exe` suffix on Windows).

This pattern appears in `safetensors_to_gguf.py`, `quantize_gguf.py`, `convert_and_quantize.py`, `analyze_gguf.py`, and `analyze_model.py`. See [Operations](operations.md) for setup guidance.

## Design Decisions from Git History

- **Removed hardcoded paths** (PR #3, `edf66dc`): Early versions had hardcoded `/Users/dave/llama.cpp` paths. The portable discovery pattern replaced them.
- **Removed always-failing MoE path** (PR #4, `c9c0511`): An earlier MoE quantization code path never worked and was removed in favor of the two-step pipeline.
- **Version-adaptive adapter** (PR #7, #10): The `UpstreamConverter` was introduced to handle `ModelBase` rename and the `conversion` package split without pinning to one llama.cpp version.
- **Full architecture support** (PR #10, `8f432ea`): Switched from `from_model_architecture` (eager registry read) to `get_model_class` (lazy import), unlocking all 234 llama.cpp architectures.
- **Multimodal projector export** (PR #11, `5ba825f`): Added `--mmproj` flag routing to llama.cpp's projector classes.
