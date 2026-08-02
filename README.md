# SafeTensors to GGUF Converter

A toolkit for working with Hugging Face models and GGUF format for use with [llama.cpp](https://github.com/ggerganov/llama.cpp). It includes tools to convert SafeTensors to GGUF and to quantize GGUF models to more efficient formats.

## Features

### SafeTensors to GGUF Conversion
- Converts SafeTensors model files to GGUF by driving llama.cpp's own converter
- **Every architecture llama.cpp supports** — 234 text architectures and 55 multimodal projectors — resolved through llama.cpp's own dispatch, not a hardcoded list
- **Mistral-native models** (`params.json` + `consolidated*.safetensors`) detected automatically
- **Multimodal projector export** with `--mmproj`, for the vision models llama.cpp supports
- **Version-adaptive**: probes the llama.cpp checkout you point it at and adapts, instead of pinning one API shape
- Automatically detects the llama.cpp directory or allows custom path specification

### GGUF Quantization
- Quantizes GGUF models to more efficient formats
- Supports various quantization types (q4_0, q4_k, q5_k, etc.)
- Automatically names output files based on quantization type
- Provides size comparison between original and quantized models
- Special handling for Mixture of Experts (MoE) models
- Model structure analysis to optimize quantization

### Two-Step Conversion and Quantization for MoE Models
- Single script that handles both conversion and quantization in one command
- Useful for MoE models, whose experts often need different quantization than the rest
- Creates uncompressed intermediate GGUF files (F16/F32) before quantization
- Solves the issue of models that are already in a compressed format

## Requirements

- **Python 3.10 or higher** — llama.cpp's converter declares `requires-python = '>=3.10,<3.15'`, and this tool imports it
- Python packages: `numpy`, `torch`, `transformers`, `safetensors`, `sentencepiece` (install via `pip install -r requirements.txt`)
- Optional: [`mistral-common`](https://pypi.org/project/mistral-common/), required only to convert Mistral-native models
- Access to llama.cpp repository (either as a parent directory or specified via command line)

> **Why `transformers`?** `safetensors_to_gguf.py` delegates the actual conversion to llama.cpp's `convert_hf_to_gguf.py`, which imports `transformers`, `safetensors`, and `sentencepiece`. If these are not installed you will see errors such as `No module named 'transformers'`, even though this repo's own scripts don't import them directly.

### llama.cpp compatibility

`safetensors_to_gguf.py` drives llama.cpp's own conversion code, which has changed shape several times. Rather than pinning one layout, the tool now **probes** for whatever the checkout it is pointed at provides, and delegates architecture resolution and class lookup to llama.cpp itself. Supported generations:

| llama.cpp | Shape | Status |
|---|---|---|
| before [#14737](https://github.com/ggml-org/llama.cpp/pull/14737) (`a3a7874`, 2025-08-11) | `load_hparams(dir_model)`; no Mistral format | supported (`--mistral-format` unavailable) |
| #14737 … before [#17114](https://github.com/ggml-org/llama.cpp/pull/17114) | `load_hparams(dir_model, is_mistral_format)`; monolithic `convert_hf_to_gguf.py` with an eagerly-populated class registry | supported |
| [#17114](https://github.com/ggml-org/llama.cpp/pull/17114) (`cc7200bf`, 2026-05-15) and later | converter split into a `conversion` package; `convert_hf_to_gguf.py` re-exports helpers only and model classes register **lazily** via `get_model_class` | supported |

Because class lookup now goes through llama.cpp's `get_model_class` (which imports the owning `conversion.<family>` module on demand) rather than a bare registry read, **every architecture llama.cpp supports is available here** — 234 text architectures and 55 multimodal projectors at the time of writing, not just Llama-family models.

- **Mistral-native models** (a `params.json` and `consolidated*.safetensors`, no `config.json`) are detected automatically and converted with llama.cpp's `MistralModel` / `MistralMoeModel`. Force the choice with `--mistral-format` / `--no-mistral-format`. This path additionally requires the [`mistral-common`](https://pypi.org/project/mistral-common/) package.
- **Multimodal projectors** are exported with `--mmproj`, which routes to llama.cpp's projector class for the architecture (`PixtralModel` for Mistral-native vision models, `Gemma3VisionModel`, `Qwen3VLVisionModel`, and so on). 42 architectures resolve to a *different* class depending on whether `--mmproj` is set, so the same model directory yields either the text model or its projector.
- **If you hit an "Incompatible llama.cpp" error**, the converter API has changed in a way this loader does not yet understand. Please open an issue with the llama.cpp commit hash. Pointing `--llama-cpp-dir` at an older checkout is still a valid workaround.

## Project Structure and File Map

This toolkit is organized as a set of standalone CLI utilities and supporting files. Below is a breakdown of each file, its purpose, and its dependencies:

| File                        | Purpose                                                                 | Key Dependencies                |
|-----------------------------|-------------------------------------------------------------------------|----------------------------------|
| `safetensors_to_gguf.py`    | Main CLI tool to convert SafeTensors models to GGUF. Probes the llama.cpp checkout and delegates architecture/class resolution to it. | Python stdlib, `numpy`, `llama.cpp` Python modules (`gguf`), access to `llama.cpp` repo |
| `convert_and_quantize.py`   | Two-step CLI tool for MoE models: converts SafeTensors to uncompressed GGUF, then quantizes. Calls `safetensors_to_gguf.py` and uses `llama-quantize` binary. | Python stdlib, `numpy`, subprocess, `llama.cpp` binaries, `safetensors_to_gguf.py` |
| `quantize_gguf.py`          | CLI tool to quantize GGUF models using `llama.cpp`'s quantization utilities. | Python stdlib, `numpy`, subprocess, `llama.cpp` binaries |
| `analyze_gguf.py`           | CLI tool to analyze GGUF model structure using the `gguf` Python module. | Python stdlib, `numpy`, `llama.cpp` Python modules (`gguf`) |
| `analyze_gguf_simple.py`    | Simpler GGUF analyzer using the `llama-quantize` binary for tensor info. | Python stdlib, `numpy`, subprocess, `llama.cpp` binaries |
| `analyze_model.py`          | Analyzes GGUF model structure, especially for MoE detection, using the `llama-quantize` binary. | Python stdlib, `numpy`, subprocess, `llama.cpp` binaries |
| `model_analysis.json`       | (Optional) Stores model analysis results. Used for reference or output. | -                                |
| `tests/test_upstream_compat.py` | Compatibility tests. Builds miniature llama.cpp checkouts for each converter generation and asserts the adapter handles all of them. Needs neither torch nor a real llama.cpp. | Python stdlib (`unittest`) |
| `requirements.txt`          | Python dependencies.                                                    | -                                |
| `LICENSE`, `.gitignore`     | Standard project files.                                                 | -                                |
| `README.md`                 | Project documentation and usage instructions.                           | -                                |

### Dependency Notes
- **llama.cpp**: Most scripts require access to the [llama.cpp](https://github.com/ggerganov/llama.cpp) repository, both for Python modules (e.g., `gguf`) and for binaries (e.g., `llama-quantize`).
- **numpy**: Used for tensor and model analysis in several scripts.
- **transformers / safetensors / sentencepiece**: Required transitively by llama.cpp's `convert_hf_to_gguf.py`, which `safetensors_to_gguf.py` loads to perform the conversion. Install them via `requirements.txt`.
- **No direct inter-script imports**: Scripts do not import each other as Python modules, but some (e.g., `convert_and_quantize.py`) may call others as subprocesses or by path.

This mapping should help users understand the role of each file and how the tools fit together in the SafeTensors-to-GGUF and quantization workflow.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/elementalcollision/safetensorstogguf.git
   cd safetensorstogguf
   ```

2. Install the Python dependencies (a virtual environment is recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have access to the llama.cpp repository. You can either:
   - Clone llama.cpp in a parent directory
   - Specify the path to llama.cpp using the `--llama-cpp-dir` parameter

## Usage

### SafeTensors to GGUF Conversion

```bash
python safetensors_to_gguf.py --model /path/to/model --outfile /path/to/output.gguf
```

### GGUF Quantization

```bash
python quantize_gguf.py --model /path/to/model.gguf --type q4_k
```

### Two-Step Conversion and Quantization for MoE Models

```bash
python convert_and_quantize.py --safetensors-dir /path/to/model --type q4_k
```

### SafeTensors to GGUF Command Line Options

- `--model`: Path to the directory containing the model's SafeTensors files (required)
- `--outfile`: Path to write the output GGUF file (default: model directory name with .gguf extension)
- `--outtype`: Output data type (default: auto)
  - Options: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto
- `--bigendian`: Use big endian format for output file (default: little endian / x86)
- `--vocab-only`: Extract only the vocabulary (mutually exclusive with `--mmproj`)
- `--mmproj`: Export the multimodal projector (vision encoder) instead of the text model. Only works on vision models llama.cpp supports for projector export; adds an `mmproj-` prefix to the default output filename
- `--model-name`: Override the model name in the GGUF file metadata
- `--metadata`: Path to a JSON file containing metadata to add to the GGUF file
- `--threads`: Number of threads to use for conversion (default: number of CPU cores)
- `--verbose`: Enable verbose logging
- `--llama-cpp-dir`: Path to the llama.cpp directory (default: auto-detect)
- `--mistral-format` / `--no-mistral-format`: Force or disable Mistral-native conversion (`params.json` + `consolidated*.safetensors`). Auto-detected by default; requires `mistral-common`
- `--optimize-for-size`, `--optimize-output-tensor`, `--optimize-token-embeddings`: **No-ops.** These set hints in `hparams` that no current llama.cpp model class reads. Retained for CLI compatibility; use `quantize_gguf.py` to actually reduce model size

## Examples

### Basic SafeTensors to GGUF Conversion

```bash
python safetensors_to_gguf.py --model /path/to/Llama-4-Scout-17B-16E-Instruct
```

### Specifying Output Format and llama.cpp Directory

```bash
python safetensors_to_gguf.py --model /path/to/Llama-4-Scout-17B-16E-Instruct --outtype f16 --llama-cpp-dir /path/to/llama.cpp
```

### Converting Only the Vocabulary

```bash
python safetensors_to_gguf.py --model /path/to/Llama-4-Scout-17B-16E-Instruct --vocab-only
```

### Converting a Mistral-Native Model

Mistral-native layouts (`params.json` + `consolidated*.safetensors`, no `config.json`) are detected automatically — no flag needed:

```bash
python safetensors_to_gguf.py --model /path/to/Ministral-3-3B-Instruct-2512
```

Repositories that ship *both* layouts default to HuggingFace format, matching llama.cpp. Force the Mistral path explicitly:

```bash
python safetensors_to_gguf.py --model /path/to/Mistral-7B-Instruct-v0.3 --mistral-format
```

### Exporting a Multimodal Projector

For vision models, the projector is a separate GGUF alongside the text model:

```bash
# text model  -> MyVisionModel.gguf
python safetensors_to_gguf.py --model /path/to/MyVisionModel

# projector   -> mmproj-MyVisionModel.gguf
python safetensors_to_gguf.py --model /path/to/MyVisionModel --mmproj
```

### Basic GGUF Quantization

```bash
python quantize_gguf.py --model /path/to/model.gguf --type q4_k
```

### GGUF Quantization with Custom Output Path

```bash
python quantize_gguf.py --model /path/to/model.gguf --type q5_k --outfile /path/to/output-q5k.gguf
```

### Model Structure Analysis (MoE Detection)

```bash
python quantize_gguf.py --model /path/to/model.gguf --analyze-model --type auto
```

### Quantizing with an Importance Matrix

An importance matrix (imatrix) tells the quantizer which weights matter most, improving quality at a given size. The low-bit `iq*` types cannot be produced without one — llama.cpp refuses them outright.

```bash
# 1. Generate the matrix from a calibration corpus (any representative text)
llama-imatrix -m model.gguf -f calibration.txt -o model.imatrix

# 2. Quantize using it
python quantize_gguf.py --model model.gguf --type iq2_xxs --imatrix model.imatrix
```

Without `--imatrix`, that `--type` is rejected up front rather than failing after the model has been loaded. To scope the matrix to particular tensors:

```bash
python quantize_gguf.py --model model.gguf --type q4_k --imatrix model.imatrix \
  --exclude-weights attn_q --exclude-weights attn_k
```

`convert_and_quantize.py` accepts the same three flags and forwards them to its quantization step.

### MoE-Specific Quantization

Experts dominate an MoE model's size, while the router is tiny and precision-sensitive — so it is usually worth quantizing them differently from each other:

```bash
python quantize_gguf.py --model /path/to/model.gguf --type q4_k \
  --moe-expert-quantization q4_k --moe-router-quantization f32
```

These map onto llama.cpp's `--tensor-type NAME=TYPE`, targeting `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` for experts and `ffn_gate_inp` for the router. Both default to `same`, which leaves those tensors to `--type`.

### Basic Two-Step Conversion and Quantization for MoE Models

```bash
python convert_and_quantize.py --safetensors-dir /path/to/Llama-4-Scout-17B-16E-Instruct --type q4_k
```

### Advanced Two-Step Conversion with Different Quantization Types

```bash
python convert_and_quantize.py --safetensors-dir /path/to/Llama-4-Scout-17B-16E-Instruct \
  --intermediate-type f32 --type q5_k --moe-expert-quantization q8_0 --moe-router-quantization f16 \
  --keep-intermediate --leave-output-tensor
```

### Complete Conversion Pipeline (Manual Method)

```bash
# Step 1: Convert SafeTensors to GGUF
python safetensors_to_gguf.py --model /path/to/Llama-4-Scout-17B-16E-Instruct

# Step 2: Quantize the resulting GGUF file
python quantize_gguf.py --model /path/to/Llama-4-Scout-17B-16E-Instruct.gguf --type q4_k
```

## Supported Models

Model support is **whatever your llama.cpp checkout supports** — the architecture is resolved by llama.cpp's own `get_model_architecture` and the class by its `get_model_class`, so there is no list in this repo to fall out of date. Against llama.cpp `fb92d8f` that is **234 text architectures across 70 modules** and **55 multimodal projectors across 26 modules**: Llama, Qwen, DeepSeek, Gemma, GLM, Granite, Mistral, Mamba, Kimi, MiniMax, Ernie, Hunyuan, Phi, and so on.

To see the catalogue your checkout offers:

```bash
python /path/to/llama.cpp/convert_hf_to_gguf.py --print-supported-models
```

End-to-end conversions verified against llama.cpp `fb92d8f`:

| model | path exercised | result |
|---|---|---|
| `DeepseekV3ForCausalLM` (tiny-random) | HF format, lazy class registry | 34 tensors, MoE experts stacked and retained |
| `Ministral-3-3B-Instruct-2512` | Mistral-native, auto-detected | 236 tensors, 6.86 GB, all hyperparameters carried through |
| `Gemma3ForConditionalGeneration` | `--mmproj` routing | resolves `Gemma3VisionModel` vs `Gemma3Model` |

## How It Works

The script drives llama.cpp's own conversion code rather than reimplementing it:

1. **Locate and probe llama.cpp** — resolve the base class (`ModelBase`, or `Model` on pre-2025 checkouts) and detect which helpers this checkout provides (`conversion` package vs monolithic module, `get_model_class`, `get_model_architecture`).
2. **Load hyperparameters** — call `load_hparams` with or without `is_mistral_format` depending on the signature actually present.
3. **Resolve the architecture** — via llama.cpp's `get_model_architecture`, which understands nested layouts (`text_config`, InternVL's `llm_config`, Qwen2.5-Omni's `thinker_config`, non-HF Mamba's `ssm_cfg`).
4. **Resolve the model class** — via `get_model_class`, which imports the owning `conversion.<family>` module on demand. This is what makes the full llama.cpp model catalogue available. With `--mmproj` the lookup is done under `ModelType.MMPROJ`, which for 42 architectures returns a different class than the text lookup.
5. **Convert** — instantiate that class and write the GGUF.

Mistral-native models skip steps 3–4: `params.json` has no `architectures` key, so the class is chosen from the payload shape exactly as llama.cpp does it (`PixtralModel` under `--mmproj`, `MistralMoeModel` when a `moe` key is present, otherwise `MistralModel`).

Llama-4 (including the MoE variants) is handled by llama.cpp's own `Llama4Model`. Earlier releases of this tool substituted a local subclass that dropped router and expert tensors; that produced files some runtimes would not load, and it has been removed.

## Testing

```bash
python3 -m unittest discover -s tests -v
```

The suite builds miniature llama.cpp checkouts reproducing each generation of the converter's public surface — one-argument monolith, two-argument monolith, and the `conversion` package — and asserts the adapter handles all three. It needs neither torch nor a real llama.cpp checkout and runs in well under a second.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the core conversion utilities
- Hugging Face for the SafeTensors format

## Current Limitations

- **Llama-4 output is unverified on real weights.** Earlier releases converted Llama-4 with a local subclass that skipped router and expert tensors, and those files did not load correctly in LM Studio. Llama-4 now goes through llama.cpp's own `Llama4Model` with all tensors intact, which should resolve it — but no full Llama-4 conversion has been run since. Reports welcome.
- **Converted files have not been load-tested in a llama.cpp runtime.** Outputs are validated structurally (read back with `GGUFReader`, metadata cross-checked against the source config), not by running inference.
- **The `--optimize-*` flags are inert.** `--optimize-for-size`, `--optimize-output-tensor` and `--optimize-token-embeddings` are passed into `hparams` but no current llama.cpp model class reads them. They are kept for CLI compatibility and do nothing.
- **Pre-#17114 llama.cpp is covered by tests, not by a real run.** The monolithic and one-argument `load_hparams` generations are exercised against stub checkouts; recent end-to-end runs all used current llama.cpp.
- **`--mmproj` requires a model llama.cpp ships a projector class for.** Anything else exits with a message telling you to drop the flag.

## GGUF Quantization Command Line Options

- `--model`: Path to the input GGUF model file (required)
- `--outfile`: Path to write the output quantized GGUF file (default: same directory as input with quantization type suffix)
- `--type`: Quantization type (default: q4_k)
  - Standard options: q4_0, q4_1, q5_0, q5_1, q8_0
  - K-quant options: q2_k, q3_k, q4_k, q5_k, q6_k
  - IQ options: iq2_xxs, iq2_xs, iq3_xxs, iq3_xs, iq4_nl
  - Full precision: f16, bf16, f32
  - Special value: `auto` — analyse the model and print quantization recommendations without writing a file (no `--analyze-model` needed)
- `--imatrix`: Path to an importance matrix from `llama-imatrix`. Improves quality at a given size, and is **required** by the low-bit types `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `q2_k_s`
- `--include-weights` / `--exclude-weights`: Apply the importance matrix to only, or to all but, the named tensor. Repeatable; mutually exclusive; require `--imatrix`
- `--threads`: Number of threads to use for quantization (default: number of CPU cores)
- `--allow-requantize`: Allow requantizing tensors that have already been quantized
- `--leave-output-tensor`: Leave output.weight unquantized (increases model size but may improve quality)
- `--pure`: Disable k-quant mixtures and quantize all tensors to the same type
- `--output-tensor-type`: ggml type for the `output.weight` tensor. Accepts any type llama-quantize knows — `f32`, `f16`, `bf16`, `q8_0`, `q6_k`, `iq4_nl`, … (the valid set is read from the `gguf` module, so it tracks llama.cpp)
- `--token-embedding-type`: ggml type for the token embeddings tensor. Same accepted set as above

### MoE-Specific Options

- `--analyze-model`: Analyze model structure before quantization to identify tensor distribution and MoE components
- `--moe-expert-quantization`: Quantization type for MoE expert weights (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`). Default `same` leaves them to `--type`
- `--moe-router-quantization`: Quantization type for the MoE router (`ffn_gate_inp`). Routers are small and sensitive, so `f32` or `f16` is often worthwhile. Default `same`
- `--verbose`: Enable verbose logging
- `--llama-cpp-dir`: Path to the llama.cpp directory (default: auto-detect)

## Two-Step Conversion and Quantization Command Line Options

- `--safetensors-dir`: Directory containing SafeTensors model files (required)
- `--outfile`: Path to write the final quantized GGUF file
- `--outdir`: Output directory for the final quantized GGUF model (if --outfile not specified)
- `--type`: Quantization type for the final model (default: q4_k)
- `--intermediate-type`: Format for the intermediate uncompressed GGUF file (f16, f32, default: f16)
- `--moe-expert-quantization`: Quantization type for MoE expert weights (default `same`)
- `--moe-router-quantization`: Quantization type for the MoE router (default `same`)
- `--llama-cpp-dir`: Path to llama.cpp directory (if not automatically detected)
- `--keep-intermediate`: Keep the intermediate uncompressed GGUF file
- `--verbose`: Enable verbose output
- `--threads`: Number of threads to use for quantization
- `--allow-requantize`: Allow requantizing tensors that are already quantized
- `--leave-output-tensor`: Leave the output tensor in the original format (f16/f32)
- `--output-tensor-type`: Output tensor type (f32, f16)
- `--token-embedding-type`: Token embedding tensor type (f32, f16)