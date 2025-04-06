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
from pathlib import Path
from typing import List, Optional, Dict, Any

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
            script_dir                 # If script is directly in llama.cpp
        ]
        
        for path in possible_paths:
            quantize_binary = path / "quantize"
            if os.name == 'nt':  # Windows
                quantize_binary = path / "quantize.exe"
                
            if quantize_binary.exists():
                llama_cpp_dir = path
                break
    
    if llama_cpp_dir is None or not (llama_cpp_dir / "quantize").exists():
        if os.name == 'nt' and not (llama_cpp_dir / "quantize.exe").exists():
            raise ValueError(
                "Could not find llama.cpp directory with quantize binary. Please specify it using --llama-cpp-dir. "
                "Make sure you have built the quantize binary."
            )
    
    return llama_cpp_dir

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
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", 
            "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k",
            "f16", "f32"
        ], default="q4_k",
        help="Quantization type (default: q4_k)"
    )
    
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads to use for quantization (default: number of CPU cores)"
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

def quantize_gguf_model(args):
    """
    Quantize a GGUF model using llama.cpp's quantize tool.
    
    Args:
        args: Command line arguments
    """
    # Verify that the model file exists
    if not args.model.exists():
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    # Set up llama.cpp path
    try:
        llama_cpp_dir = setup_llama_cpp_path(args.llama_cpp_dir)
        logger.info(f"Using llama.cpp directory: {llama_cpp_dir}")
    except Exception as e:
        logger.error(f"Error setting up llama.cpp path: {e}")
        return 1
    
    # Determine the quantize binary path
    quantize_binary = llama_cpp_dir / "quantize"
    if os.name == 'nt':  # Windows
        quantize_binary = llama_cpp_dir / "quantize.exe"
    
    # Ensure the quantize binary exists
    if not quantize_binary.exists():
        logger.error(f"Quantize binary not found at {quantize_binary}")
        logger.error("Make sure you have built llama.cpp with the quantize target")
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
    
    # Build the command
    cmd = [str(quantize_binary), str(args.model), str(outfile), args.type]
    
    # Add threads if specified
    if args.threads:
        cmd.extend(["-t", str(args.threads)])
    
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
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                logger.info(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            logger.info(f"Quantization completed successfully. Output file: {outfile}")
            logger.info(f"Original size: {args.model.stat().st_size / (1024 * 1024):.2f} MB")
            logger.info(f"Quantized size: {outfile.stat().st_size / (1024 * 1024):.2f} MB")
            return 0
        else:
            logger.error(f"Quantization failed with return code {return_code}")
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
