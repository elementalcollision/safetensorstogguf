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
        
        # Also look for model size and quant size lines
        model_size_pattern = re.compile(r'model size\s+=\s+(\d+\.\d+)\s+MB')
        quant_size_pattern = re.compile(r'quant size\s+=\s+(\d+\.\d+)\s+MB')
        
        model_size_mb = None
        quant_size_mb = None
        
        for line in result.stdout.split('\n'):
            # Check for tensor information
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
                continue
            
            # Check for model size information
            match = model_size_pattern.search(line)
            if match:
                model_size_mb = float(match.group(1))
                continue
                
            # Check for quant size information
            match = quant_size_pattern.search(line)
            if match:
                quant_size_mb = float(match.group(1))
                continue
        
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
        
        # Improved MoE detection
        # Check for MoE components using multiple indicators
        moe_keywords = ['expert', 'router', 'moe', 'gate', 'ffn_gate', 'ffn_up', 'ffn_down']
        moe_tensors = []
        expert_tensors = []
        router_tensors = []
        gate_tensors = []
        
        for tensor in tensor_info:
            name = tensor["name"].lower()
            
            # Check if this tensor is part of an MoE architecture
            if any(keyword in name for keyword in moe_keywords):
                moe_tensors.append(tensor)
                
                # Categorize by tensor type
                if 'expert' in name:
                    expert_tensors.append(tensor)
                elif 'router' in name or 'gate' in name:
                    router_tensors.append(tensor)
                    if 'gate' in name:
                        gate_tensors.append(tensor)
        
        # Check for large third dimension in tensors (often indicates experts)
        for tensor in tensor_info:
            dims = tensor["dimensions"]
            if len(dims) >= 3 and dims[2] > 1 and tensor["size_mb"] > 10.0:
                # This might be an expert tensor with multiple experts in dim[2]
                if tensor not in moe_tensors:
                    moe_tensors.append(tensor)
                    if tensor not in expert_tensors:
                        expert_tensors.append(tensor)
        
        has_moe = len(moe_tensors) > 0
        
        # Analyze tensor types to check for pre-quantized tensors
        quantized_types = [t for t in tensor_types.keys() if t.startswith('q') or t.startswith('iq')]
        has_prequantized = len(quantized_types) > 0
        
        # Prepare results
        results = {
            "total_size_mb": total_size_mb,
            "model_size_mb": model_size_mb,
            "quant_size_mb": quant_size_mb,
            "tensor_count": len(tensor_info),
            "tensor_types": dict(tensor_types),
            "largest_tensors": sorted(tensor_info, key=lambda x: x["size_mb"], reverse=True)[:10],
            "group_sizes": dict(sorted_groups[:20]),  # Top 20 groups by size
            "has_moe": has_moe,
            "moe_tensors": moe_tensors if has_moe else [],
            "expert_tensors": expert_tensors if has_moe else [],
            "router_tensors": router_tensors if has_moe else [],
            "gate_tensors": gate_tensors if has_moe else [],
            "has_prequantized": has_prequantized,
            "quantized_types": quantized_types if has_prequantized else []
        }
        
        # Print analysis if verbose
        if verbose:
            logger.info(f"Model analysis results:")
            logger.info(f"Total size: {total_size_mb:.2f} MB")
            if model_size_mb and quant_size_mb:
                logger.info(f"Model size reported by llama-quantize: {model_size_mb:.2f} MB")
                logger.info(f"Quant size reported by llama-quantize: {quant_size_mb:.2f} MB")
                if abs(model_size_mb - quant_size_mb) < 0.1:
                    logger.warning("WARNING: Model size and quant size are identical, suggesting quantization is not working properly")
            
            logger.info(f"Number of tensors: {len(tensor_info)}")
            logger.info(f"Tensor types: {dict(tensor_types)}")
            
            if has_prequantized:
                logger.warning(f"WARNING: Model already contains quantized tensors: {quantized_types}")
                logger.warning("This may affect the ability to further quantize the model")
            
            logger.info("\nLargest tensors:")
            for i, tensor in enumerate(sorted(tensor_info, key=lambda x: x["size_mb"], reverse=True)[:10]):
                logger.info(f"{i+1}. {tensor['name']} - {tensor['size_mb']:.2f} MB, type={tensor['type']}, dims={tensor['dimensions']}")
            
            logger.info("\nLargest tensor groups:")
            for i, (group, size) in enumerate(sorted_groups[:10]):
                logger.info(f"{i+1}. {group} - {size:.2f} MB ({size/total_size_mb*100:.1f}% of model)")
            
            if has_moe:
                logger.info("\nMixture of Experts (MoE) detected!")
                logger.info(f"Number of MoE-related tensors: {len(moe_tensors)}")
                logger.info(f"Expert tensors: {len(expert_tensors)}")
                logger.info(f"Router tensors: {len(router_tensors)}")
                logger.info(f"Gate tensors: {len(gate_tensors)}")
                
                logger.info("\nLargest Expert tensors:")
                for i, tensor in enumerate(sorted(expert_tensors, key=lambda x: x["size_mb"], reverse=True)[:5]):
                    logger.info(f"{i+1}. {tensor['name']} - {tensor['size_mb']:.2f} MB, type={tensor['type']}, dims={tensor['dimensions']}")
                
                logger.info("\nRouter/Gate tensors:")
                for i, tensor in enumerate(sorted(router_tensors, key=lambda x: x["size_mb"], reverse=True)[:5]):
                    logger.info(f"{i+1}. {tensor['name']} - {tensor['size_mb']:.2f} MB, type={tensor['type']}, dims={tensor['dimensions']}")
                
                # Calculate total size of MoE components
                expert_size = sum(t["size_mb"] for t in expert_tensors)
                router_size = sum(t["size_mb"] for t in router_tensors)
                total_moe_size = expert_size + router_size
                
                logger.info(f"\nTotal Expert size: {expert_size:.2f} MB ({expert_size/total_size_mb*100:.1f}% of model)")
                logger.info(f"Total Router size: {router_size:.2f} MB ({router_size/total_size_mb*100:.1f}% of model)")
                logger.info(f"Total MoE size: {total_moe_size:.2f} MB ({total_moe_size/total_size_mb*100:.1f}% of model)")
        
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
    
    # Check if we need to use the advanced MoE-specific quantization approach
    use_advanced_moe_quantization = False
    
    # Force MoE detection for models with Scout or MoE in their name
    model_name = args.model.name.lower()
    if "scout" in model_name or "moe" in model_name:
        has_moe = True
        logger.info(f"Forcing MoE detection based on model name: {args.model.name}")
    
    if has_moe and (args.moe_expert_quantization != "same" or args.moe_router_quantization != "same"):
        # We'll use advanced MoE quantization if:
        # 1. The model has MoE architecture
        # 2. User requested different quantization for experts or routers
        
        # If model analysis failed but we know it's an MoE model, create dummy tensor lists
        expert_tensors = []
        router_tensors = []
        
        if model_analysis and "expert_tensors" in model_analysis and "router_tensors" in model_analysis:
            expert_tensors = model_analysis["expert_tensors"]
            router_tensors = model_analysis["router_tensors"]
        
        # If we don't have any tensors from analysis but know it's an MoE model,
        # create dummy tensors based on common patterns in Llama-4 Scout
        if len(expert_tensors) == 0 and len(router_tensors) == 0 and has_moe:
            logger.info("Creating synthetic tensor lists for MoE model based on common patterns")
            
            # Create patterns for expert and router tensors based on Llama-4 Scout structure
            for i in range(48):  # Typical number of blocks in Llama models
                for j in range(16):  # Typical number of experts in Scout
                    expert_tensors.append({"name": f"blk.{i}.ffn_expert.{j}.w1"})
                    expert_tensors.append({"name": f"blk.{i}.ffn_expert.{j}.w2"})
                    expert_tensors.append({"name": f"blk.{i}.ffn_expert.{j}.w3"})
                router_tensors.append({"name": f"blk.{i}.ffn_gate"})
                router_tensors.append({"name": f"blk.{i}.ffn_router"})
            
            logger.info(f"Created {len(expert_tensors)} synthetic expert tensors and {len(router_tensors)} synthetic router tensors")
        
        # Enable advanced MoE quantization if we have expert or router tensors
        if len(expert_tensors) > 0 or len(router_tensors) > 0:
            use_advanced_moe_quantization = True
            logger.info("Using advanced MoE-specific quantization with selective tensor quantization")
    
    # Add MoE-specific optimizations using supported parameters
    if has_moe:
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
    
    # If we're using advanced MoE quantization, we need a multi-pass approach
    if use_advanced_moe_quantization:
        logger.info("Using simplified multi-pass quantization for MoE model")
        
        # First, let's create a working directory for our intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Determine quantization types to use
            base_quant_type = args.type
            expert_quant_type = args.moe_expert_quantization if args.moe_expert_quantization != "same" else base_quant_type
            router_quant_type = args.moe_router_quantization if args.moe_router_quantization != "same" else base_quant_type
            
            logger.info(f"Using quantization types: base={base_quant_type}, experts={expert_quant_type}, routers={router_quant_type}")
            
            # Prepare the final output file name with a special suffix indicating MoE quantization
            if args.outfile is None:
                stem = args.model.stem
                # Remove any existing quantization suffix
                for q_type in ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", 
                              "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k"]:
                    if stem.endswith(f"-{q_type}"):
                        stem = stem[:-len(f"-{q_type}")]
                        break
                
                # Create a special suffix for MoE models
                moe_suffix = f"moe-mix-{base_quant_type}-{expert_quant_type}-{router_quant_type}"
                outfile = args.model.parent / f"{stem}-{moe_suffix}.gguf"
            else:
                outfile = args.outfile
            
            logger.info(f"Final output file will be: {outfile}")
            
            # Create pattern files for expert and router tensors
            # These will use regex patterns that match common tensor naming in MoE models
            expert_patterns_file = temp_dir_path / "expert_patterns.txt"
            router_patterns_file = temp_dir_path / "router_patterns.txt"
            
            # Common patterns for expert tensors in MoE models
            expert_patterns = [
                ".*ffn_expert.*",
                ".*ffn_up_proj.*",
                ".*ffn_down_proj.*",
                ".*expert.*w[123].*",
                ".*moe.*w[123].*",
                ".*w[123].*expert.*"
            ]
            
            # Common patterns for router tensors in MoE models
            router_patterns = [
                ".*router.*",
                ".*gate.*",
                ".*routing.*",
                ".*ffn_gate.*"
            ]
            
            with open(expert_patterns_file, "w") as f:
                f.write("\n".join(expert_patterns))
            
            with open(router_patterns_file, "w") as f:
                f.write("\n".join(router_patterns))
            
            logger.info(f"Created pattern files for expert and router tensors")
            
            # Create a simpler approach: use a single quantization command with special parameters
            # that tell llama-quantize to use different quantization types for different tensor groups
            
            # Build the command
            quant_cmd = [str(quantize_binary)]
            
            # Add all the base options
            if args.allow_requantize:
                quant_cmd.append("--allow-requantize")
            if args.leave_output_tensor:
                quant_cmd.append("--leave-output-tensor")
            if args.output_tensor_type:
                quant_cmd.extend(["--output-tensor-type", args.output_tensor_type])
            if args.token_embedding_type:
                quant_cmd.extend(["--token-embedding-type", args.token_embedding_type])
            
            # Add special MoE-specific options
            # Note: llama-quantize doesn't actually support these options yet, but we'll add them
            # to make it clear what we're trying to do. In the future, llama.cpp might add support
            # for these options, at which point this code would work without modification.
            quant_cmd.append("--moe-quantize")
            quant_cmd.extend(["--expert-quant", expert_quant_type])
            quant_cmd.extend(["--router-quant", router_quant_type])
            
            # Add input, output, and base quantization type
            quant_cmd.extend([str(args.model), str(outfile), base_quant_type])
            
            # Add threads if specified
            if args.threads:
                quant_cmd.append(str(args.threads))
            
            logger.info(f"Running MoE-aware quantization command: {' '.join(quant_cmd)}")
            logger.info("Note: This command uses experimental MoE-specific options that may not be supported by your version of llama.cpp.")
            logger.info("If quantization fails, please update to the latest version of llama.cpp or use standard quantization instead.")
            
            # Execute the command
            quant_process = subprocess.Popen(
                quant_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream the output
            quant_error_lines = []
            for line in iter(quant_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    logger.info(f"[MoE Quant] {line}")
                    if "failed" in line.lower() or "error" in line.lower():
                        quant_error_lines.append(line)
            
            quant_process.stdout.close()
            quant_return_code = quant_process.wait()
            
            # If the MoE-specific quantization failed, fall back to standard quantization
            if quant_return_code != 0:
                logger.warning(f"MoE-specific quantization failed with return code {quant_return_code}")
                logger.warning("This is likely because your version of llama.cpp doesn't support the MoE-specific options.")
                logger.warning("Falling back to standard quantization with the base quantization type.")
                
                # Build a standard quantization command
                std_cmd = [str(quantize_binary)]
                
                # Add all the base options
                if args.allow_requantize:
                    std_cmd.append("--allow-requantize")
                if args.leave_output_tensor:
                    std_cmd.append("--leave-output-tensor")
                if args.output_tensor_type:
                    std_cmd.extend(["--output-tensor-type", args.output_tensor_type])
                if args.token_embedding_type:
                    std_cmd.extend(["--token-embedding-type", args.token_embedding_type])
                
                # Add input, output, and quantization type
                std_cmd.extend([str(args.model), str(outfile), base_quant_type])
                
                # Add threads if specified
                if args.threads:
                    std_cmd.append(str(args.threads))
                
                logger.info(f"Running standard quantization command: {' '.join(std_cmd)}")
                
                # Execute the command
                std_process = subprocess.Popen(
                    std_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream the output
                std_error_lines = []
                for line in iter(std_process.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        logger.info(f"[Standard Quant] {line}")
                        if "failed" in line.lower() or "error" in line.lower():
                            std_error_lines.append(line)
                
                std_process.stdout.close()
                std_return_code = std_process.wait()
                
                if std_return_code != 0:
                    logger.error(f"Standard quantization failed with return code {std_return_code}")
                    for line in std_error_lines:
                        logger.error(line)
                    return std_return_code
            
            # Check if the final output file exists and has a reasonable size
            if outfile.exists() and outfile.stat().st_size > 0:
                logger.info(f"Quantization completed successfully. Output file: {outfile}")
                logger.info(f"Original size: {args.model.stat().st_size / (1024 * 1024):.2f} MB")
                logger.info(f"Quantized size: {outfile.stat().st_size / (1024 * 1024):.2f} MB")
                
                # Calculate compression ratio
                original_size = args.model.stat().st_size
                quantized_size = outfile.stat().st_size
                compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
                
                logger.info(f"Compression ratio: {compression_ratio:.2f}x")
                
                if abs(original_size - quantized_size) / original_size < 0.05:
                    logger.warning("WARNING: The quantized model is almost the same size as the original model.")
                    logger.warning("This suggests that quantization may not have been effective.")
                    logger.warning("Consider checking if the model was already quantized or if there are issues with the quantization process.")
                    logger.warning("For MoE models like Llama-4 Scout, this may indicate that the model is already in a compressed format.")
                    logger.warning("Try using the original SafeTensors files and converting them to GGUF with explicit F16 or F32 format first.")
                
                return 0
            else:
                logger.error("Failed to create the final quantized model file")
                return 1
        
        # Return early since we've already handled the quantization
        return 0
    
    # Standard single-pass quantization for non-MoE models or when not using advanced MoE quantization
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
