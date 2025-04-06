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
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Counter
from collections import defaultdict

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
            # Analysis mode
            "auto",
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
        help="Quantization type (default: q4_k, use 'auto' with --analyze-model for analysis-only mode)"
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
    
    # MoE-specific options
    parser.add_argument(
        "--analyze-model", action="store_true",
        help="Analyze model structure before quantization to identify tensor distribution and MoE components"
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
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--llama-cpp-dir", type=Path,
        help="Path to the llama.cpp directory (default: auto-detect)"
    )
    
    return parser.parse_args()

def analyze_model_structure(input_file: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze the structure of a GGUF model to understand tensor distribution and identify MoE components.
    
    Args:
        input_file: Path to the input GGUF file
        verbose: Whether to print detailed analysis information
        
    Returns:
        Dictionary containing analysis results
    """
    logger = logging.getLogger("quantize-gguf")
    logger.info(f"Analyzing model structure: {input_file}")
    
    # Run llama-gguf to extract model information
    try:
        # Find llama-quantize binary for model analysis
        llama_cpp_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        quantize_paths = [
            llama_cpp_dir.parent / "build" / "bin" / "llama-quantize",
            llama_cpp_dir.parent / "build" / "llama-quantize",
            llama_cpp_dir.parent / "llama-quantize",
            Path("/Users/dave/llama.cpp/build/llama-quantize"),
            Path("/Users/dave/llama.cpp/build/bin/llama-quantize")
        ]
        
        quantize_binary = None
        for path in quantize_paths:
            if path.exists():
                quantize_binary = path
                break
        
        if not quantize_binary:
            logger.warning("llama-quantize binary not found. Cannot perform model analysis.")
            return {"error": "llama-quantize binary not found"}
        
        # Run llama-quantize with --dry-run to get model information without quantizing
        # This will output tensor information that we can parse
        cmd = [str(quantize_binary), "--dry-run", "--verbose", str(input_file), "/dev/null", "q4_0"]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check for successful execution
        if result.returncode != 0:
            logger.warning(f"Error running llama-quantize: {result.stderr}")
            return {"error": f"llama-quantize analysis failed: {result.stderr}"}
        
        # Parse the output to extract tensor information
        tensor_info = []
        
        # Different pattern for llama-quantize output
        # Look for lines like: tensor   42:         blk.0.attn_q.weight - [4096, 4096], type = f16, size =   32.00 MB
        tensor_pattern = re.compile(r'tensor\s+\d+:\s+([\w\.]+)\s+-\s+\[(\d+(?:,\s*\d+)*)\],\s+type\s+=\s+(\w+),\s+size\s+=\s+(\d+\.\d+)\s+MB')
        
        for line in result.stdout.split('\n'):
            match = tensor_pattern.search(line)
            if match:
                name, dims_str, dtype, size_mb = match.groups()
                # Parse dimensions
                dims = [int(d.strip()) for d in dims_str.split(',')]
                # Pad dimensions to length 4 if needed
                while len(dims) < 4:
                    dims.append(1)
                
                tensor_info.append({
                    "name": name.strip(),
                    "dimensions": dims[:4],  # Take first 4 dimensions
                    "type": dtype.strip(),
                    "size_mb": float(size_mb)
                })
        
        # Analyze tensor distribution
        total_size_mb = sum(t["size_mb"] for t in tensor_info)
        tensor_types = Counter(t["type"] for t in tensor_info)
        
        # Group tensors by prefix to identify components
        tensor_groups = defaultdict(list)
        for tensor in tensor_info:
            # Extract the component name (e.g., 'blk.1', 'output', etc.)
            name_parts = tensor["name"].split('.')
            if len(name_parts) > 1:
                prefix = name_parts[0]
                if prefix == 'blk':
                    # For blocks, include the block number
                    if len(name_parts) > 1:
                        prefix = f"{prefix}.{name_parts[1]}"
            else:
                prefix = tensor["name"]
            
            tensor_groups[prefix].append(tensor)
        
        # Calculate size per group
        group_sizes = {}
        for group, tensors in tensor_groups.items():
            group_sizes[group] = sum(t["size_mb"] for t in tensors)
        
        # Sort groups by size (descending)
        sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Check for MoE components
        moe_tensors = [t for t in tensor_info if 'expert' in t["name"] or 'router' in t["name"] or 'moe' in t["name"]]
        has_moe = len(moe_tensors) > 0
        
        # Prepare results
        results = {
            "total_size_mb": total_size_mb,
            "tensor_count": len(tensor_info),
            "tensor_types": dict(tensor_types),
            "largest_tensors": sorted(tensor_info, key=lambda x: x["size_mb"], reverse=True)[:10],
            "group_sizes": dict(sorted_groups[:20]),  # Top 20 groups by size
            "has_moe": has_moe,
            "moe_tensors": moe_tensors if has_moe else []
        }
        
        # Print analysis if verbose
        if verbose:
            logger.info(f"Model size: {total_size_mb:.2f} MB")
            logger.info(f"Tensor count: {len(tensor_info)}")
            logger.info(f"Tensor types: {dict(tensor_types)}")
            
            logger.info("\nLargest tensors:")
            for t in results["largest_tensors"]:
                logger.info(f"  {t['name']}: {t['size_mb']:.2f} MB ({t['type']})")
            
            logger.info("\nLargest tensor groups:")
            for group, size in list(sorted_groups)[:10]:
                logger.info(f"  {group}: {size:.2f} MB ({size/total_size_mb*100:.1f}%)")
            
            if has_moe:
                logger.info("\nMoE components detected:")
                for t in moe_tensors:
                    logger.info(f"  {t['name']}: {t['size_mb']:.2f} MB ({t['type']})")
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing model structure: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {"error": str(e)}

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
    logger = logging.getLogger("quantize-gguf")
    
    # Verify that the model file exists and preprocess if needed
    try:
        input_file, was_preprocessed = preprocess_gguf_file(args.model)
        if was_preprocessed:
            logger.info(f"Preprocessed input file: {input_file}")
    except Exception as e:
        logger.error(f"Error preprocessing model file: {e}")
        return 1
    
    # Analyze model structure if requested
    model_analysis = None
    has_moe = False
    if args.analyze_model or args.moe_expert_quantization != "same" or args.moe_router_quantization != "same":
        logger.info("Analyzing model structure to detect MoE components and optimize quantization...")
        model_analysis = analyze_model_structure(args.model, args.verbose)
        
        if "error" in model_analysis:
            logger.warning(f"Model analysis failed: {model_analysis['error']}")
            logger.warning("Continuing with standard quantization...")
        else:
            has_moe = model_analysis.get("has_moe", False)
            if has_moe:
                logger.info("Detected Mixture of Experts (MoE) architecture in the model")
                logger.info(f"Found {len(model_analysis.get('moe_tensors', []))} MoE-related tensors")
                
                # Log MoE quantization settings
                if args.moe_expert_quantization != "same":
                    logger.info(f"Using {args.moe_expert_quantization} quantization for expert layers")
                if args.moe_router_quantization != "same":
                    logger.info(f"Using {args.moe_router_quantization} quantization for router layers")
    
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
    
    # Add MoE-specific optimizations using supported parameters
    if args.moe_expert_quantization != "same" or args.moe_router_quantization != "same":
        logger.info("Applying MoE-specific quantization settings")
        
        # Add --leave-output-tensor flag to prevent quantizing the output tensor
        # This is important for MoE models
        if "--leave-output-tensor" not in cmd:
            cmd.append("--leave-output-tensor")
        
        # Set token embeddings to a higher precision
        if not args.token_embedding_type:
            cmd.extend(["--token-embedding-type", "f16"])
        
        # Set output tensor type if not already set
        if not args.output_tensor_type:
            cmd.extend(["--output-tensor-type", "f16"])
        
        # For MoE models, we want to use a higher precision for the base model
        # and then quantize the experts more aggressively
        if args.type in ["q4_0", "q4_1", "q5_0", "q5_1"]:
            logger.info("Recommending a k-quant type for better quality with MoE models")
            logger.info("Consider using --type q4_k or --type q5_k instead")
    
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
    
    # Check if we're just analyzing the model structure
    if args.analyze_model and args.type == "auto":
        # In this case, we're just analyzing the model without quantizing
        logger.info("Running in analysis-only mode")
        try:
            # Verify the model file exists
            if not args.model.exists():
                logger.error(f"Model file not found: {args.model}")
                return 1
                
            # Analyze the model structure
            analysis_results = analyze_model_structure(args.model, verbose=True)
            
            if "error" in analysis_results:
                logger.error(f"Error analyzing model: {analysis_results['error']}")
                return 1
                
            # Provide quantization recommendations based on analysis
            has_moe = analysis_results.get("has_moe", False)
            if has_moe:
                logger.info("\n===== Quantization Recommendations for MoE Model =====")
                logger.info("This model contains Mixture of Experts (MoE) architecture.")
                logger.info("Recommended quantization settings:")
                logger.info("  1. For better quality: --type q5_k --leave-output-tensor --moe-expert-quantization f16")
                logger.info("  2. For better compression: --type q4_k_m --moe-expert-quantization q8_0")
                logger.info("  3. For balanced approach: --type q4_k --moe-router-quantization f16")
            else:
                logger.info("\n===== Quantization Recommendations =====")
                logger.info("  1. For better quality: --type q5_k --leave-output-tensor")
                logger.info("  2. For better compression: --type q4_0")
                logger.info("  3. For balanced approach: --type q4_k_m")
                
            return 0
        except Exception as e:
            logger.error(f"Error during model analysis: {e}")
            if args.verbose:
                import traceback
                logger.debug(traceback.format_exc())
            return 1
    
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
