---
type: Workflows
title: Workflows — Conversion, Quantization, and MoE Pipeline
description: The three core CLI workflows (SafeTensors→GGUF conversion, GGUF quantization with MoE support, and the two-step convert+quantize pipeline), plus model analysis. Includes CLI reference for all flags.
tags: [workflows, conversion, quantization, moe, cli-reference, analysis]
---

# Workflows

Three core workflows plus an analysis mode. Each is a standalone CLI script.

## 1. SafeTensors → GGUF Conversion

**Script:** `safetensors_to_gguf.py`

```bash
python safetensors_to_gguf.py --model /path/to/model --outfile output.gguf
```

### How it works

1. `setup_llama_cpp_path()` probes the llama.cpp checkout and builds an `UpstreamConverter` (see [Architecture](architecture.md)).
2. `resolve_mistral_format()` auto-detects Mistral-native layout (`params.json` + no `config.json`).
3. `UPSTREAM.load_hparams()` loads hyperparameters.
4. `UPSTREAM.model_architecture()` resolves the architecture string via llama.cpp's own resolver.
5. `UPSTREAM.model_class()` resolves the model class (lazy import on #17114+).
6. The model class is instantiated and `.write()` (or `.write_vocab()` for `--vocab-only`) is called under `torch.inference_mode()`.

### Mistral-native models

Detected automatically when the directory has `params.json` and no `config.json`. Force with `--mistral-format` / `--no-mistral-format`. Requires the `mistral-common` package.

### Multimodal projector export (`--mmproj`)

Routes to a different model class (e.g. `Gemma3VisionModel` vs `Gemma3Model`). 42 architectures resolve to a different class depending on whether `--mmproj` is set. Adds `mmproj-` prefix to the default output filename. Mutually exclusive with `--vocab-only`.

### CLI flags

| Flag | Description |
|---|---|
| `--model` | Path to model directory (required) |
| `--outfile` | Output GGUF path (default: model dir / `<name>.gguf`) |
| `--outtype` | f32, f16, bf16, q8_0, tq1_0, tq2_0, auto (default: auto) |
| `--bigendian` | Big-endian output |
| `--vocab-only` | Extract vocabulary only |
| `--mmproj` | Export multimodal projector |
| `--model-name` | Override model name in metadata |
| `--metadata` | JSON file with metadata to add |
| `--threads` | Thread count |
| `--verbose` | Debug logging |
| `--llama-cpp-dir` | Path to llama.cpp |
| `--mistral-format` / `--no-mistral-format` | Force/disable Mistral-native |
| `--optimize-for-size`, `--optimize-output-tensor`, `--optimize-token-embeddings` | **No-ops.** Retained for CLI compatibility. |

## 2. GGUF Quantization

**Script:** `quantize_gguf.py`

```bash
python quantize_gguf.py --model model.gguf --type q4_k
```

### How it works

1. Optional `--analyze-model` runs `analyze_model_structure()` which reads tensors directly from the GGUF file via `GGUFReader` (not `llama-quantize --dry-run`, which was the previous broken approach).
2. `setup_llama_cpp_path()` locates the `llama-quantize` binary.
3. Output path defaults to `<stem>-<type>.gguf` in the input's directory.
4. Builds the `llama-quantize` command with flags, then runs it as a subprocess with streamed output.

### MoE-aware quantization

MoE expert and router tensors can be targeted independently via `--moe-expert-quantization` and `--moe-router-quantization`. These map to `llama-quantize`'s `--tensor-type NAME=TYPE` (repeatable):

- **Expert tensors:** `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`
- **Router tensors:** `ffn_gate_inp`

The function `moe_tensor_type_args(expert_type, router_type)` builds the flag list. `same` means leave to `--type`.

```bash
python quantize_gguf.py --model model.gguf --type q4_k \
  --moe-expert-quantization q4_k --moe-router-quantization f32
```

When MoE is detected (or the model name contains "scout" or "moe"), the script also auto-applies `--leave-output-tensor`, `--token-embedding-type f16`, and `--output-tensor-type f16` if not explicitly set.

### Analysis-only mode

```bash
python quantize_gguf.py --model model.gguf --analyze-model --type auto
```

Analyzes tensor distribution, detects MoE components, checks for pre-quantized tensors, and prints quantization recommendations. Does not quantize.

### CLI flags

| Flag | Description |
|---|---|
| `--model` | Input GGUF file (required) |
| `--outfile` | Output path (default: `<stem>-<type>.gguf`) |
| `--type` | Quantization type (default: q4_k). `auto` for analysis-only. |
| `--threads` | Thread count |
| `--allow-requantize` | Allow re-quantizing already-quantized tensors |
| `--leave-output-tensor` | Leave output.weight unquantized |
| `--pure` | Disable k-quant mixtures |
| `--output-tensor-type` | Type for output.weight (f32, f16, q8_0, q4_0, q4_1) |
| `--token-embedding-type` | Type for token embeddings |
| `--analyze-model` | Analyze before quantizing |
| `--moe-expert-quantization` | Type for expert tensors (default: same) |
| `--moe-router-quantization` | Type for router (default: same) |
| `--verbose` | Debug logging |
| `--llama-cpp-dir` | Path to llama.cpp |

### Supported quantization types

- **Standard:** q4_0, q4_1, q5_0, q5_1, q8_0
- **K-quant:** q2_k, q2_k_s, q3_k (s/m/l), q4_k (s/m), q5_k (s/m), q6_k
- **IQ:** iq1_s, iq1_m, iq2_xxs/xs/s/m, iq3_xxs/xs/s/m, iq4_nl/xs
- **Ternary:** tq1_0, tq2_0
- **Full precision:** f16, bf16, f32

## 3. Two-Step Convert + Quantize (MoE Pipeline)

**Script:** `convert_and_quantize.py`

```bash
python convert_and_quantize.py --safetensors-dir /path/to/model --type q4_k
```

### How it works

1. `setup_llama_cpp_path()` locates the converter script (sibling `safetensors_to_gguf.py`) and the `llama-quantize` binary.
2. `convert_safetensors_to_gguf()` runs `safetensors_to_gguf.py` as a subprocess, outputting to an intermediate file in f16 or f32 format. Uses a temp directory unless `--keep-intermediate`.
3. `quantize_gguf_model()` runs `llama-quantize` on the intermediate file to produce the final quantized output.
4. Cleans up the temp directory unless `--keep-intermediate`.

This solves the problem of MoE models shipped already-compressed: the intermediate F16/F32 file is always uncompressed, so `llama-quantize` can work on it.

### Caveat: MoE flags in convert_and_quantize.py

`convert_and_quantize.py` accepts `--moe-expert-quantization` / `--moe-router-quantization` but currently **warns and ignores them** (line ~287). Unlike `quantize_gguf.py`, it does not yet map them to `--tensor-type`. Use `quantize_gguf.py` directly if you need per-tensor MoE targeting.

### CLI flags

| Flag | Description |
|---|---|
| `--safetensors-dir` | Model directory (required) |
| `--outfile` | Final output path |
| `--outdir` | Output directory |
| `--type` | Final quantization type (default: q4_k) |
| `--intermediate-type` | Intermediate format: f16 or f32 (default: f16) |
| `--moe-expert-quantization` | Accepted but currently ignored with warning |
| `--moe-router-quantization` | Accepted but currently ignored with warning |
| `--llama-cpp-dir` | Path to llama.cpp |
| `--keep-intermediate` | Keep the intermediate GGUF |
| `--verbose` / `-v` | Verbose output |
| `--threads` | Thread count |
| `--allow-requantize` | Allow requantization |
| `--leave-output-tensor` | Leave output tensor unquantized |
| `--output-tensor-type` | f32 or f16 |
| `--token-embedding-type` | f32 or f16 |

## 4. Model Analysis Tools

Three standalone analysis scripts exist alongside the built-in `--analyze-model` in `quantize_gguf.py`:

- **`analyze_gguf.py`** — Uses the `gguf` Python module's `GGUFReader` to print metadata, tensor types, shapes, and sizes. No binary required.
- **`analyze_gguf_simple.py`** — Uses `llama-quantize` binary for tensor info. Simpler output format.
- **`analyze_model.py`** — MoE-focused analyzer using `llama-quantize --dry-run`. Note: the `--dry-run --verbose` approach used here is known to be unreliable (see [Testing](testing.md) for the regression context); the `analyze_model_structure()` in `quantize_gguf.py` is the recommended replacement.

## Relationship to Architecture and Operations

The conversion workflow depends on the [UpstreamConverter adapter](architecture.md) for all llama.cpp interactions. The quantization and two-step workflows depend on [llama.cpp binary discovery](operations.md) for `llama-quantize`. When troubleshooting either, check the operations page for known limitations and error patterns.
