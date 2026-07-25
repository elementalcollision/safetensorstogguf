---
type: Project Overview
title: SafeTensors to GGUF — Quickstart
description: Entry point for the safetensorstogguf toolkit. Covers what the tools do, how to install them, quick-start commands, and links to architecture, workflows, operations, and testing documentation.
tags: [quickstart, gguf, safetensors, llama-cpp, conversion, quantization]
---

# SafeTensors to GGUF Toolkit

A set of standalone Python CLI utilities for converting Hugging Face SafeTensors models to GGUF format and quantizing them for use with [llama.cpp](https://github.com/ggerganov/llama.cpp). The toolkit delegates conversion and quantization to llama.cpp's own code rather than reimplementing it, adapting to multiple generations of the llama.cpp converter API.

## What It Does

1. **Convert** SafeTensors → GGUF by driving llama.cpp's `convert_hf_to_gguf.py`, supporting every architecture llama.cpp supports (234 text, 55 multimodal projectors).
2. **Quantize** GGUF models to compact formats (q4_k, q5_k, iq4_xs, etc.) via the `llama-quantize` binary, with MoE-aware per-tensor targeting.
3. **Two-step pipeline** for MoE models: convert to uncompressed GGUF first, then quantize — solving the problem of models already shipped in a compressed format.
4. **Analyze** GGUF model structure to detect MoE components, tensor distributions, and pre-quantized weights.

## Install

```bash
git clone https://github.com/elementalcollision/safetensorstogguf.git
cd safetensorstogguf
pip install -r requirements.txt
```

You also need a llama.cpp checkout with `convert_hf_to_gguf.py` and the `llama-quantize` binary built. See [Operations](operations.md) for discovery details and troubleshooting.

## Quick Commands

```bash
# Convert SafeTensors to GGUF
python safetensors_to_gguf.py --model /path/to/model --outfile output.gguf

# Quantize a GGUF model
python quantize_gguf.py --model model.gguf --type q4_k

# Two-step convert + quantize (for MoE models)
python convert_and_quantize.py --safetensors-dir /path/to/model --type q4_k

# Analyze a GGUF model's structure
python quantize_gguf.py --model model.gguf --analyze-model --type auto
```

## Documentation Sections

- [Architecture](architecture.md) — How the adapter pattern works, the `UpstreamConverter` class, llama.cpp version compatibility, and file map.
- [Workflows](workflows.md) — The three core workflows (conversion, quantization, two-step pipeline) plus model analysis, with CLI reference.
- [Operations](operations.md) — Setup, llama.cpp discovery, troubleshooting, known limitations, and regenerating this wiki.
- [Testing](testing.md) — Test suite structure, how to run tests, and what each test guards against.

## File Map at a Glance

| File | Role |
|---|---|
| `safetensors_to_gguf.py` | Main converter CLI. Probes llama.cpp, delegates architecture resolution. |
| `quantize_gguf.py` | Quantization CLI with MoE analysis and per-tensor targeting. |
| `convert_and_quantize.py` | Two-step pipeline: convert then quantize in one command. |
| `analyze_gguf.py` | GGUF structure analyzer using the `gguf` Python module. |
| `analyze_gguf_simple.py` | Simpler analyzer using the `llama-quantize` binary. |
| `analyze_model.py` | MoE-focused model analyzer using `llama-quantize --dry-run`. |
| `tests/test_upstream_compat.py` | Adapter compatibility tests across llama.cpp generations. |
| `tests/test_quantize.py` | Quantization regression tests (MoE tensor targeting, analysis). |

## Backlog

- **`analyze_gguf_simple.py` and `analyze_model.py`** — Older analysis tools that predate the `read_gguf_tensors` / `analyze_model_structure` path in `quantize_gguf.py`. Still functional but not covered in workflow docs in detail; consider consolidating in a future update.
- **`model_analysis.json`** — A 17-byte placeholder file. No meaningful content to document.
