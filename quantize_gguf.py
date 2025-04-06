#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_gguf.py - A CLI tool to quantize GGUF models using llama.cpp

This tool leverages llama.cpp's quantization utilities to convert GGUF models
to more efficient quantized formats.
"""

import argparse
import logging
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Configure logging
logger = logging.getLogger("quantize-gguf")
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
            quantize_binary = path / "llama-quantize"
            if os.name == 'nt':  # Windows
                quantize_binary = path / "llama-quantize.exe"
                
            if quantize_binary.exists():
                llama_cpp_dir = path
                logger.info(f"Found llama.cpp directory at: {llama_cpp_dir}")
                break
                
            # Also check in the build/bin directory
            build_bin_quantize_binary = path / "build" / "bin" / "llama-quantize"
            if os.name == 'nt':  # Windows
                build_bin_quantize_binary = path / "build" / "bin" / "llama-quantize.exe"
                
            if build_bin_quantize_binary.exists():
                llama_cpp_dir = path
                logger.info(f"Found llama.cpp directory at: {llama_cpp_dir} (build/bin directory)")
                break
    
    if llama_cpp_dir is None:
        raise ValueError(
            "Could not find llama.cpp directory with llama-quantize binary. Please specify it using --llama-cpp-dir. "
            "Make sure you have built the llama-quantize binary using 'cmake .. && make llama-quantize'."
        )
    
    # Check if the quantize binary exists in the specified directory or build directories
    quantize_binary = llama_cpp_dir / "llama-quantize"
    build_quantize_binary = llama_cpp_dir / "build" / "llama-quantize"
    build_bin_quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    
    if os.name == 'nt':  # Windows
        quantize_binary = llama_cpp_dir / "llama-quantize.exe"
        build_quantize_binary = llama_cpp_dir / "build" / "llama-quantize.exe"
        build_bin_quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize.exe"
    
    if quantize_binary.exists():
        return llama_cpp_dir, quantize_binary
    elif build_quantize_binary.exists():
        return llama_cpp_dir, build_quantize_binary
    elif build_bin_quantize_binary.exists():
        return llama_cpp_dir, build_bin_quantize_binary
    else:
        raise ValueError(
            f"The llama-quantize binary was not found in the specified llama.cpp directory: {llama_cpp_dir} "
            "or its build subdirectories. Make sure you have built the llama-quantize binary "
            "using 'cmake .. && make llama-quantize'."
        )
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quantize GGUF models to more efficient formats using llama.cpp"
    )
    
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to the input GGUF model file"
    )
    
    parser.add_argument(
        "--outfile", type=Path,
        help="Path to write the output quantized GGUF file (default: same directory as input with quantization type suffix)"
    )
    
    parser.add_argument(
        "--type", type=str, choices=[
            # Standard quantization types
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", 
            # K-quant types (better quality)
            "q2_k", "q2_k_s", "q3_k", "q3_k_s", "q3_k_m", "q3_k_l", 
            "q4_k", "q4_k_s", "q4_k_m", "q5_k", "q5_k_s", "q5_k_m", "q6_k",
            # IQ types (best compression)
            "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
            "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m",
            "iq4_nl", "iq4_xs",
            "iq1_s", "iq1_m",
            # Ternary quantization
            "tq1_0", "tq2_0",
            # Full precision
            "f16", "bf16", "f32"
        ], default="q4_k",
        help="Quantization type (default: q4_k)"
    )
    
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads to use for quantization (default: number of CPU cores)"
    )
    
    parser.add_argument(
        "--allow-requantize", action="store_true",
        help="Allow requantizing tensors that have already been quantized (may reduce quality)"
    )
    
    parser.add_argument(
        "--leave-output-tensor", action="store_true",
        help="Leave output.weight unquantized. Increases model size but may improve quality"
    )
    
    parser.add_argument(
        "--pure", action="store_true",
        help="Disable k-quant mixtures and quantize all tensors to the same type"
    )
    
    parser.add_argument(
        "--output-tensor-type", type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1"],
        help="Use this type for the output.weight tensor"
    )
    
    parser.add_argument(
        "--token-embedding-type", type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1"],
        help="Use this type for the token embeddings tensor"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--llama-cpp-dir", type=Path,
        help="Path to the llama.cpp directory (default: auto-detect)"
    )
    
    return parser.parse_args()

def preprocess_gguf_file(input_file: Path) -> Tuple[Path, bool]:
    """
    Preprocess the GGUF file to add any missing keys required by the quantization tool.
    
    Args:
        input_file: Path to the input GGUF file
        
    Returns:
        Tuple of (processed_file_path, was_modified)
    """
    # For now, we'll just check if the file exists and return it as-is
    # In the future, we could add preprocessing steps here if needed
    if not input_file.exists():
        raise FileNotFoundError(f"Model file not found: {input_file}")
    
    # We'll return the original file for now
    # If we need to modify it in the future, we can create a temporary copy
    return input_file, False

def quantize_gguf_model(args):
    """
    Quantize a GGUF model using llama.cpp's quantize tool.
    
    Args:
        args: Command line arguments
    """
    # Verify that the model file exists and preprocess if needed
    try:
        input_file, was_preprocessed = preprocess_gguf_file(args.model)
        if was_preprocessed:
            logger.info(f"Preprocessed input file: {input_file}")
    except Exception as e:
        logger.error(f"Error preprocessing model file: {e}")
        return 1
    
    # Set up llama.cpp path
    try:
        llama_cpp_dir, quantize_binary = setup_llama_cpp_path(args.llama_cpp_dir)
        logger.info(f"Using llama.cpp directory: {llama_cpp_dir}")
        logger.info(f"Using quantize binary: {quantize_binary}")
    except Exception as e:
        logger.error(f"Error setting up llama.cpp path: {e}")
        return 1
    
    # Ensure the quantize binary exists
    if not quantize_binary.exists():
        logger.error(f"Quantize binary not found at {quantize_binary}")
        logger.error("Make sure you have built llama.cpp with the llama-quantize target")
        return 1
    
    # Determine output file path if not specified
    outfile = args.outfile
    if outfile is None:
        # Generate output filename based on input and quantization type
        stem = args.model.stem
        # Remove any existing quantization suffix
        for q_type in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", 
                      "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k"]:
            if stem.endswith(f"-{q_type}"):
                stem = stem[:-len(f"-{q_type}")]
                break
        
        outfile = args.model.parent / f"{stem}-{args.type}.gguf"
        logger.info(f"No output file specified, using: {outfile}")
    
    # Build the command - llama-quantize expects options BEFORE the input/output files
    cmd = [str(quantize_binary)]
    
    # Add optional flags first
    if args.allow_requantize:
        cmd.append("--allow-requantize")
        
    if args.leave_output_tensor:
        cmd.append("--leave-output-tensor")
        
    if args.pure:
        cmd.append("--pure")
        
    if args.output_tensor_type:
        cmd.extend(["--output-tensor-type", args.output_tensor_type])
        
    if args.token_embedding_type:
        cmd.extend(["--token-embedding-type", args.token_embedding_type])
    
    # Add input file, output file, and quantization type
    cmd.extend([str(args.model), str(outfile), args.type])
    
    # Add threads as the last parameter if specified
    if args.threads:
        cmd.append(str(args.threads))
    
    # Execute the command
    logger.info(f"Running quantization command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream the output
        error_lines = []
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                logger.info(line)
                if "failed" in line.lower() or "error" in line.lower():
                    error_lines.append(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            logger.info(f"Quantization completed successfully. Output file: {outfile}")
            logger.info(f"Original size: {args.model.stat().st_size / (1024 * 1024):.2f} MB")
            logger.info(f"Quantized size: {outfile.stat().st_size / (1024 * 1024):.2f} MB")
            return 0
        else:
            logger.error(f"Quantization failed with return code {return_code}")
            
            # Check for specific error patterns and provide helpful messages
            if any("key not found in model: llama.attention.layer_norm_rms_epsilon" in line for line in error_lines):
                logger.error("\nError: Missing 'llama.attention.layer_norm_rms_epsilon' key in the GGUF file.")
                logger.error("This is likely because the GGUF file was created with an older version of llama.cpp.")
                logger.error("\nPossible solutions:")
                logger.error("1. Update your safetensors_to_gguf.py script to add this parameter during conversion")
                logger.error("2. Use an older version of llama.cpp's quantize tool that's compatible with your GGUF file")
                logger.error("3. Regenerate the GGUF file with the latest version of llama.cpp")
            
            return return_code
    
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        return 1

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("quantize-gguf")
    
    # Quantize the model
    try:
        return quantize_gguf_model(args)
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
