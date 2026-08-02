---
type: Operations
title: Operations — Setup, Troubleshooting, and Known Limitations
description: How to set up llama.cpp for the toolkit, how the discovery pattern works, common errors and their fixes, known limitations, and how to regenerate this wiki.
tags: [operations, setup, troubleshooting, limitations, llama-cpp-discovery]
---

# Operations

## Prerequisites

- **Python 3.10+** (llama.cpp's converter requires `>=3.10,<3.15`)
- **llama.cpp checkout** with `convert_hf_to_gguf.py` present and `llama-quantize` binary built
- Python packages: `numpy`, `torch`, `transformers`, `safetensors`, `sentencepiece` (install via `pip install -r requirements.txt`)
- Optional: `mistral-common` for Mistral-native model conversion
- Optional: `gguf` package (or use llama.cpp's bundled `gguf-py`)

## Building llama.cpp

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
```

The `llama-quantize` binary will be at `build/bin/llama-quantize` (Linux/macOS) or `build/bin/llama-quantize.exe` (Windows).

## llama.cpp Discovery

All scripts use the same discovery order:

1. `--llama-cpp-dir` CLI flag (highest priority)
2. `LLAMA_CPP_DIR` environment variable
3. Relative to the script: `parent.parent`, `parent`, `script_dir` itself

The same order locates the bundled `gguf-py` when the `gguf` package is not installed, so `--llama-cpp-dir` alone is sufficient — an explicit flag takes precedence over `LLAMA_CPP_DIR`.

The converter (`safetensors_to_gguf.py`) looks for `convert_hf_to_gguf.py` in the llama.cpp root. The quantizer and analysis tools look for `llama-quantize` in the root, `build/`, or `build/bin/`.

If discovery fails:

```
Could not find llama.cpp directory. Please specify it using --llama-cpp-dir.
```

**Fix:** Pass `--llama-cpp-dir /path/to/llama.cpp` or set `LLAMA_CPP_DIR`.

## Common Errors

### "No module named 'transformers'"

`safetensors_to_gguf.py` doesn't import `transformers` directly, but llama.cpp's `convert_hf_to_gguf.py` does. Install all transitive deps:

```bash
pip install -r requirements.txt
```

### "Incompatible llama.cpp: could not find a `ModelBase` (or legacy `Model`) class"

The llama.cpp checkout is too old or too new in a way the adapter doesn't understand. Check that the checkout has `convert_hf_to_gguf.py` with a `ModelBase` or `Model` class. If it's a new layout the adapter hasn't seen, open an issue with the commit hash.

### "Missing 'llama.attention.layer_norm_rms_epsilon' key"

The GGUF file was created with an older converter that didn't add this metadata key. Regenerate the GGUF with the current `safetensors_to_gguf.py`, or use an older `llama-quantize` that doesn't require it.

### "Quantized model is almost the same size as the original"

The model may already be in a compressed format. Use `convert_and_quantize.py` to produce an uncompressed intermediate first, or check with `--analyze-model` for pre-quantized tensor types.

### Importance matrix (imatrix)

`--imatrix`, `--include-weights` and `--exclude-weights` are forwarded to `llama-quantize` by both `quantize_gguf.py` and `convert_and_quantize.py`.

Seven quantization types cannot be produced without one — `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs` and `q2_k_s`. llama.cpp raises `this quantization requires an imatrix!` for them, but only after loading the model (`tensor_requires_imatrix()` in `src/llama-quant.cpp`); the wrappers reject the combination at argument-parse time instead. Note `token_embd` and `output.weight` are exempt upstream, and `iq4_nl`, `iq4_xs`, `iq3_s` and the k-quants work fine without one.

Generate a matrix with `llama-imatrix -m model.gguf -f calibration.txt -o model.imatrix`. `llama-imatrix` needs a causal language model; it aborts on encoder models such as BERT.

### MoE quantization flags

`--moe-expert-quantization` / `--moe-router-quantization` work in both `quantize_gguf.py` and `convert_and_quantize.py`. Both map onto `llama-quantize`'s repeatable `--tensor-type NAME=TYPE`, targeting `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` for experts and `ffn_gate_inp` for the router. `convert_and_quantize.py` imports the mapping from `quantize_gguf.py` so the two drivers cannot diverge. The default `same` emits nothing and leaves those tensors to `--type`.

## Known Limitations

From the README and source code:

1. **Llama-4 output is unverified on real weights.** Earlier releases used a local subclass that dropped router/expert tensors; Llama-4 now goes through llama.cpp's own `Llama4Model` with all tensors intact, but no full conversion has been verified since.
2. **Converted files have not been load-tested in a llama.cpp runtime.** Outputs are validated structurally (GGUFReader read-back, metadata cross-check), not by running inference.
3. **`--optimize-*` flags are inert.** `--optimize-for-size`, `--optimize-output-tensor`, `--optimize-token-embeddings` set hparams hints that no current llama.cpp model class reads. Retained for CLI compatibility.
4. **Pre-#17114 llama.cpp is covered by tests, not by a real run.** The monolithic and one-argument `load_hparams` generations are exercised against stub checkouts in the test suite; recent end-to-end runs all used current llama.cpp.
5. **`--mmproj` requires a model llama.cpp ships a projector class for.** If the architecture has no projector class, the tool exits with a message telling you to drop the flag.
6. **Analysis output is structural, not predictive.** `analyze_model.py` and `quantize_gguf.py --analyze-model` both read the GGUF directly via `GGUFReader`, so they report the *current* tensor layout, types and sizes. Neither predicts the post-quantization size; use `llama-quantize --dry-run` for that.

## Regenerating this wiki

These pages are regenerated **manually**. There is no scheduled job and no GitHub Actions workflow in this repository — documentation is refreshed on demand by a maintainer:

```bash
npm install --global openwiki
export OPENROUTER_API_KEY=...        # provider credentials
openwiki code --update --print
```

That rewrites `openwiki/`, `AGENTS.md` and `CLAUDE.md` in the working tree; review the diff and commit it like any other change. `openwiki/.last-update.json` records the git HEAD the pages were generated against, which is the quickest way to tell whether the wiki has drifted from the code.

## Environment Variables

| Variable | Purpose |
|---|---|
| `LLAMA_CPP_DIR` | Path to llama.cpp checkout (alternative to `--llama-cpp-dir`) |
| `OPENROUTER_API_KEY` | Required when regenerating this wiki with OpenWiki |
| `LANGSMITH_API_KEY` | Optional OpenWiki tracing |
