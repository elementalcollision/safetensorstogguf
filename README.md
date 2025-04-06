# SafeTensors to GGUF Converter

A command-line tool to convert Hugging Face models in SafeTensors format to GGUF format for use with [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Features

- Converts SafeTensors model files to GGUF format
- Supports Llama-4 models with Mixture of Experts (MoE) architecture
- Handles multimodal models by skipping vision components
- Supports custom tokenizer formats used in Llama-4
- Automatically detects the llama.cpp directory or allows custom path specification

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

```bash
python safetensors_to_gguf.py --model /path/to/model --outfile /path/to/output.gguf
```

### Command Line Options

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

### Basic Conversion

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
