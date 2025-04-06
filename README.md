# SafeTensors to GGUF Converter

A toolkit for working with Hugging Face models and GGUF format for use with [llama.cpp](https://github.com/ggerganov/llama.cpp). It includes tools to convert SafeTensors to GGUF and to quantize GGUF models to more efficient formats.

## Features

### SafeTensors to GGUF Conversion
- Converts SafeTensors model files to GGUF format
- Supports Llama-4 models with Mixture of Experts (MoE) architecture
- Handles multimodal models by skipping vision components
- Supports custom tokenizer formats used in Llama-4
- Automatically detects the llama.cpp directory or allows custom path specification

### GGUF Quantization
- Quantizes GGUF models to more efficient formats
- Supports various quantization types (q4_0, q4_k, q5_k, etc.)
- Automatically names output files based on quantization type
- Provides size comparison between original and quantized models
- Special handling for Mixture of Experts (MoE) models
- Model structure analysis to optimize quantization

## Requirements

- Python 3.8 or higher
- PyTorch
- Access to llama.cpp repository (either as a parent directory or specified via command line)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/safetensors-to-gguf.git
   cd safetensors-to-gguf
   ```

2. Make sure you have access to the llama.cpp repository. You can either:
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

### SafeTensors to GGUF Command Line Options

- `--model`: Path to the directory containing the model's SafeTensors files (required)
- `--outfile`: Path to write the output GGUF file (default: model directory name with .gguf extension)
- `--outtype`: Output data type (default: auto)
  - Options: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto
- `--bigendian`: Use big endian format for output file (default: little endian / x86)
- `--vocab-only`: Extract only the vocabulary
- `--model-name`: Override the model name in the GGUF file metadata
- `--metadata`: Path to a JSON file containing metadata to add to the GGUF file
- `--threads`: Number of threads to use for conversion (default: number of CPU cores)
- `--verbose`: Enable verbose logging
- `--llama-cpp-dir`: Path to the llama.cpp directory (default: auto-detect)

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

### MoE-Specific Quantization

```bash
python quantize_gguf.py --model /path/to/model.gguf --type q4_k --moe-expert-quantization f16 --moe-router-quantization q8_0
```

### Complete Conversion Pipeline

```bash
# Step 1: Convert SafeTensors to GGUF
python safetensors_to_gguf.py --model /path/to/Llama-4-Scout-17B-16E-Instruct

# Step 2: Quantize the resulting GGUF file
python quantize_gguf.py --model /path/to/Llama-4-Scout-17B-16E-Instruct.gguf --type q4_k
```

## Supported Models

This tool has been tested with:
- Llama-4 models (including the Mixture of Experts variants)
- Other models supported by llama.cpp's convert_hf_to_gguf.py

## How It Works

The script leverages llama.cpp's conversion utilities to handle the conversion process. It adds special handling for Llama-4 models, including:

1. Support for the Mixture of Experts (MoE) architecture by skipping router weights and expert layers
2. Custom tokenizer handling for the new tokenizer format used in Llama-4
3. Proper handling of nested configuration parameters in Llama-4 models
4. Skipping multimodal components for vision-language models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the core conversion utilities
- Hugging Face for the SafeTensors format

## Current Limitations

There seems to be some issues appearing with loading the Llama-4 model after conversion. This was tested on LM Studio and will require some additional work to ensure it functions correctly. Future updates will address these compatibility issues.

## GGUF Quantization Command Line Options

- `--model`: Path to the input GGUF model file (required)
- `--outfile`: Path to write the output quantized GGUF file (default: same directory as input with quantization type suffix)
- `--type`: Quantization type (default: q4_k)
  - Standard options: q4_0, q4_1, q5_0, q5_1, q8_0
  - K-quant options: q2_k, q3_k, q4_k, q5_k, q6_k
  - IQ options: iq2_xxs, iq2_xs, iq3_xxs, iq3_xs, iq4_nl
  - Full precision: f16, bf16, f32
  - Special value: `auto` (use with `--analyze-model` for analysis-only mode)
- `--threads`: Number of threads to use for quantization (default: number of CPU cores)
- `--allow-requantize`: Allow requantizing tensors that have already been quantized
- `--leave-output-tensor`: Leave output.weight unquantized (increases model size but may improve quality)
- `--pure`: Disable k-quant mixtures and quantize all tensors to the same type
- `--output-tensor-type`: Use this type for the output.weight tensor (f32, f16, q8_0, q4_0, q4_1)
- `--token-embedding-type`: Use this type for the token embeddings tensor (f32, f16, q8_0, q4_0, q4_1)

### MoE-Specific Options

- `--analyze-model`: Analyze model structure before quantization to identify tensor distribution and MoE components
- `--moe-expert-quantization`: Quantization type for MoE expert layers (f32, f16, q8_0, q4_0, q4_1, q5_k, q4_k, same)
- `--moe-router-quantization`: Quantization type for MoE router layers (f32, f16, q8_0, q4_0, q4_1, q5_k, q4_k, same)
- `--verbose`: Enable verbose logging
- `--llama-cpp-dir`: Path to the llama.cpp directory (default: auto-detect)