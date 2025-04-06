#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_and_quantize.py - A two-step tool for MoE models that first converts SafeTensors to uncompressed GGUF,
then applies quantization to the uncompressed GGUF file.

This approach addresses the challenge with Llama-4 Scout and other MoE models that may already be in a
compressed format that can't be further quantized using standard llama.cpp quantization tools.
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Configure logging
logger = logging.getLogger("convert-and-quantize")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def setup_llama_cpp_path(llama_cpp_dir=None):
    """Set up the llama.cpp path"""
    # If not provided, try to auto-detect
    if llama_cpp_dir is None:
        # Try to find it relative to the script location
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            script_dir.parent.parent,  # If script is in llama.cpp/some_dir/safetensors-to-gguf
            script_dir.parent,         # If script is in llama.cpp/safetensors-to-gguf
            script_dir,                # If script is directly in llama.cpp
            Path("/Users/dave/llama.cpp")  # Direct path to llama.cpp
        ]
        
        for path in possible_paths:
            # Check for the binary in the main directory
            convert_binary = path / "convert-safetensors-to-gguf.py"
            quantize_binary = path / "build" / "bin" / "llama-quantize"
                
            if convert_binary.exists() and quantize_binary.exists():
                llama_cpp_dir = path
                logger.info(f"Found llama.cpp directory at: {llama_cpp_dir}")
                break
    
    if llama_cpp_dir is None:
        raise ValueError(
            "Could not find llama.cpp directory with convert-safetensors-to-gguf.py script and llama-quantize binary. "
            "Please specify it using --llama-cpp-dir."
        )
    
    # Check if the convert script and quantize binary exist
    convert_script = llama_cpp_dir / "convert-safetensors-to-gguf.py"
    quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    
    if not convert_script.exists():
        raise ValueError(f"convert-safetensors-to-gguf.py script not found at {convert_script}")
    
    if not quantize_binary.exists():
        raise ValueError(f"llama-quantize binary not found at {quantize_binary}")
    
    return llama_cpp_dir, convert_script, quantize_binary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-step tool for MoE models: first converts SafeTensors to uncompressed GGUF, then quantizes"
    )
    
    parser.add_argument(
        "--safetensors-dir", type=Path, required=True,
        help="Directory containing SafeTensors model files"
    )
    
    parser.add_argument(
        "--outfile", type=Path,
        help="Output file path for the final quantized GGUF model"
    )
    
    parser.add_argument(
        "--outdir", type=Path,
        help="Output directory for the final quantized GGUF model (if --outfile not specified)"
    )
    
    parser.add_argument(
        "--type", type=str, default="q4_k",
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k", "f16", "f32"],
        help="Quantization type for the final model"
    )
    
    parser.add_argument(
        "--intermediate-type", type=str, default="f16",
        choices=["f16", "f32"],
        help="Format for the intermediate uncompressed GGUF file"
    )
    
    parser.add_argument(
        "--moe-expert-quantization", type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_k", "q4_k", "same"],
        default="same",
        help="Quantization type for MoE expert layers (default: same as main quantization)"
    )
    
    parser.add_argument(
        "--moe-router-quantization", type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_k", "q4_k", "same"],
        default="same",
        help="Quantization type for MoE router layers (default: same as main quantization)"
    )
    
    parser.add_argument(
        "--llama-cpp-dir", type=Path,
        help="Path to llama.cpp directory (if not automatically detected)"
    )
    
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep the intermediate uncompressed GGUF file"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--threads", type=int,
        help="Number of threads to use for quantization"
    )
    
    # Add other quantization options from quantize_gguf.py
    parser.add_argument(
        "--allow-requantize", action="store_true",
        help="Allow requantizing tensors that are already quantized"
    )
    
    parser.add_argument(
        "--leave-output-tensor", action="store_true",
        help="Leave the output tensor in the original format (f16/f32)"
    )
    
    parser.add_argument(
        "--output-tensor-type", type=str,
        choices=["f32", "f16"],
        help="Output tensor type (default: unchanged)"
    )
    
    parser.add_argument(
        "--token-embedding-type", type=str,
        choices=["f32", "f16"],
        help="Token embedding tensor type (default: unchanged)"
    )
    
    return parser.parse_args()

def convert_safetensors_to_gguf(args, llama_cpp_dir, convert_script):
    """
    Convert SafeTensors model to uncompressed GGUF format.
    
    Args:
        args: Command line arguments
        llama_cpp_dir: Path to llama.cpp directory
        convert_script: Path to convert-safetensors-to-gguf.py script
        
    Returns:
        Path to the generated uncompressed GGUF file
    """
    logger.info(f"Converting SafeTensors model to uncompressed GGUF ({args.intermediate_type})...")
    
    # Determine model name from the safetensors directory
    model_name = args.safetensors_dir.name
    
    # Create a temporary directory for the intermediate file if not keeping it
    if args.keep_intermediate:
        if args.outdir:
            intermediate_dir = args.outdir
        else:
            intermediate_dir = args.safetensors_dir.parent
        
        intermediate_file = intermediate_dir / f"{model_name}-{args.intermediate_type}.gguf"
    else:
        temp_dir = tempfile.mkdtemp(prefix="llama_convert_")
        intermediate_dir = Path(temp_dir)
        intermediate_file = intermediate_dir / f"{model_name}-{args.intermediate_type}.gguf"
    
    # Build the conversion command
    cmd = [
        sys.executable,
        str(convert_script),
        "--outfile", str(intermediate_file),
        "--outtype", args.intermediate_type,
        str(args.safetensors_dir)
    ]
    
    # Add verbose flag if specified
    if args.verbose:
        cmd.append("--verbose")
    
    logger.info(f"Running conversion command: {' '.join(cmd)}")
    
    # Execute the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream the output
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            logger.info(f"[Convert] {line}")
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        logger.error(f"Conversion failed with return code {return_code}")
        if not args.keep_intermediate and 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return None
    
    logger.info(f"Successfully converted SafeTensors to uncompressed GGUF: {intermediate_file}")
    return intermediate_file, None if args.keep_intermediate else temp_dir

def quantize_gguf_model(args, intermediate_file, llama_cpp_dir, quantize_binary):
    """
    Quantize the uncompressed GGUF model.
    
    Args:
        args: Command line arguments
        intermediate_file: Path to the uncompressed GGUF file
        llama_cpp_dir: Path to llama.cpp directory
        quantize_binary: Path to llama-quantize binary
        
    Returns:
        Return code from the quantization process
    """
    logger.info(f"Quantizing uncompressed GGUF model to {args.type}...")
    
    # Determine output file path
    if args.outfile:
        outfile = args.outfile
    else:
        model_name = intermediate_file.stem
        # Remove any existing format suffix
        if model_name.endswith("-f16") or model_name.endswith("-f32"):
            model_name = model_name[:-4]
        
        if args.outdir:
            outdir = args.outdir
        else:
            outdir = intermediate_file.parent
        
        outfile = outdir / f"{model_name}-{args.type}.gguf"
    
    # Build the quantization command
    cmd = [str(quantize_binary)]
    
    # Add optional flags
    if args.allow_requantize:
        cmd.append("--allow-requantize")
    
    if args.leave_output_tensor:
        cmd.append("--leave-output-tensor")
    
    if args.output_tensor_type:
        cmd.extend(["--output-tensor-type", args.output_tensor_type])
    
    if args.token_embedding_type:
        cmd.extend(["--token-embedding-type", args.token_embedding_type])
    
    # Add MoE-specific options if specified
    if args.moe_expert_quantization != "same":
        # Note: These options are experimental and may not be supported by all llama.cpp versions
        cmd.append("--moe-quantize")
        cmd.extend(["--expert-quant", args.moe_expert_quantization])
    
    if args.moe_router_quantization != "same":
        # Note: These options are experimental and may not be supported by all llama.cpp versions
        if "--moe-quantize" not in cmd:
            cmd.append("--moe-quantize")
        cmd.extend(["--router-quant", args.moe_router_quantization])
    
    # Add input file, output file, and quantization type
    cmd.extend([str(intermediate_file), str(outfile), args.type])
    
    # Add threads if specified
    if args.threads:
        cmd.append(str(args.threads))
    
    logger.info(f"Running quantization command: {' '.join(cmd)}")
    
    # Execute the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream the output
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            logger.info(f"[Quantize] {line}")
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        logger.error(f"Quantization failed with return code {return_code}")
    else:
        logger.info(f"Successfully quantized GGUF model: {outfile}")
        
        # Calculate compression ratio
        original_size = intermediate_file.stat().st_size / (1024 * 1024)
        quantized_size = outfile.stat().st_size / (1024 * 1024)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Quantized size: {quantized_size:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        if abs(original_size - quantized_size) / original_size < 0.05:
            logger.warning("WARNING: The quantized model is almost the same size as the original model.")
            logger.warning("This suggests that quantization may not have been effective.")
            logger.warning("Check if the model was already quantized or if there are issues with the quantization process.")
    
    return return_code

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)
    
    try:
        # Set up llama.cpp path
        llama_cpp_dir, convert_script, quantize_binary = setup_llama_cpp_path(args.llama_cpp_dir)
        
        # Step 1: Convert SafeTensors to uncompressed GGUF
        result = convert_safetensors_to_gguf(args, llama_cpp_dir, convert_script)
        if not result:
            return 1
        
        intermediate_file, temp_dir = result
        
        # Step 2: Quantize the uncompressed GGUF model
        return_code = quantize_gguf_model(args, intermediate_file, llama_cpp_dir, quantize_binary)
        
        # Clean up temporary directory if created
        if temp_dir:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        
        return return_code
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
