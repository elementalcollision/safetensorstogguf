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
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter, defaultdict

# Logging is configured via logging.basicConfig() in main(); fetch the named
# logger here without attaching a second handler (which would double every line).
logger = logging.getLogger("quantize-gguf")

def setup_llama_cpp_path(llama_cpp_dir=None):
    """Set up the llama.cpp path"""
    # If not provided, try to auto-detect
    if llama_cpp_dir is None:
        # Try to find it relative to the script location
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = []
        if os.environ.get("LLAMA_CPP_DIR"):
            possible_paths.append(Path(os.environ["LLAMA_CPP_DIR"]))
        possible_paths.extend([
            script_dir.parent.parent,  # If script is in llama.cpp/some_dir/safetensors-to-gguf
            script_dir.parent,         # If script is in llama.cpp/safetensors-to-gguf
            script_dir,                # If script is directly in llama.cpp
        ])

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
        help="Quantization type (default: q4_k). Use 'auto' to analyse the model and print quantization recommendations without writing anything"
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
        "--output-tensor-type", type=str, metavar="GGML_TYPE",
        help="Use this ggml type for the output.weight tensor. Accepts any type "
             "llama-quantize knows (f32, f16, bf16, q8_0, q6_k, iq4_nl, ...)"
    )
    
    parser.add_argument(
        "--token-embedding-type", type=str, metavar="GGML_TYPE",
        help="Use this ggml type for the token embeddings tensor. Accepts any "
             "type llama-quantize knows (f32, f16, bf16, q8_0, q6_k, iq4_nl, ...)"
    )

    # Importance matrix
    parser.add_argument(
        "--imatrix", type=Path,
        help="Importance matrix produced by llama-imatrix. Improves quality at a "
             "given size, and is REQUIRED by the low-bit types: "
             + ", ".join(sorted(IMATRIX_REQUIRED_TYPES))
    )

    parser.add_argument(
        "--include-weights", action="append", metavar="TENSOR",
        help="Use the importance matrix only for this tensor. Repeatable. "
             "Requires --imatrix; mutually exclusive with --exclude-weights"
    )

    parser.add_argument(
        "--exclude-weights", action="append", metavar="TENSOR",
        help="Do not use the importance matrix for this tensor. Repeatable. "
             "Requires --imatrix; mutually exclusive with --include-weights"
    )

    # MoE-specific options
    parser.add_argument(
        "--analyze-model", action="store_true",
        help="Analyze model structure before quantizing, to report tensor distribution and MoE components. Combine with --type auto to analyse only"
    )
    
    parser.add_argument(
        "--moe-expert-quantization", type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_k", "q4_k", "same"],
        default="same",
        help="Quantization type for MoE expert layers (NOTE: not supported by upstream llama-quantize; currently ignored)"
    )

    parser.add_argument(
        "--moe-router-quantization", type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_k", "q4_k", "same"],
        default="same",
        help="Quantization type for MoE router layers (NOTE: not supported by upstream llama-quantize; currently ignored)"
    )
    
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--llama-cpp-dir", type=Path,
        help="Path to the llama.cpp directory (default: auto-detect)"
    )
    
    args = parser.parse_args()
    validate_imatrix_args(parser.error, args)
    return args


def validate_imatrix_args(fail, args) -> None:
    """Reject impossible importance-matrix combinations before doing any work.

    `fail` is a callable that reports a usage error (argparse's parser.error).
    """
    imatrix = getattr(args, "imatrix", None)
    include = getattr(args, "include_weights", None)
    exclude = getattr(args, "exclude_weights", None)

    if include and exclude:
        fail("--include-weights and --exclude-weights cannot be used together")

    if (include or exclude) and not imatrix:
        fail("--include-weights/--exclude-weights require --imatrix")

    if imatrix is not None and not Path(imatrix).is_file():
        fail(f"importance matrix not found: {imatrix}")

    quant_type = (getattr(args, "type", "") or "").lower()
    if quant_type in IMATRIX_REQUIRED_TYPES and imatrix is None:
        fail(
            f"--type {quant_type} requires an importance matrix; pass --imatrix FILE. "
            f"Generate one with llama.cpp's llama-imatrix, e.g. "
            f"`llama-imatrix -m model.gguf -f calibration.txt -o imatrix.gguf`"
        )


def imatrix_args(args) -> List[str]:
    """Build the llama-quantize importance-matrix arguments."""
    if getattr(args, "imatrix", None) is None:
        return []
    result = ["--imatrix", str(args.imatrix)]
    for tensor in getattr(args, "include_weights", None) or []:
        result.extend(["--include-weights", tensor])
    for tensor in getattr(args, "exclude_weights", None) or []:
        result.extend(["--exclude-weights", tensor])
    return result

def _add_gguf_to_path(llama_cpp_dir=None):
    """Make llama.cpp's `gguf` module importable.

    Search order, highest priority first:
      1. an already-importable `gguf` package
      2. ``llama_cpp_dir`` - normally whatever --llama-cpp-dir was given
      3. the LLAMA_CPP_DIR environment variable
      4. directories relative to this script

    An explicit --llama-cpp-dir beats the environment: the flag is the more
    specific instruction, and previously it was ignored here entirely, so
    pointing the tool at a checkout still failed with "set LLAMA_CPP_DIR".
    """
    try:
        import gguf  # noqa: F401  - already importable
        return
    except ImportError:
        pass

    candidate_dirs = []
    if llama_cpp_dir:
        candidate_dirs.append(Path(llama_cpp_dir))
    if os.environ.get("LLAMA_CPP_DIR"):
        candidate_dirs.append(Path(os.environ["LLAMA_CPP_DIR"]))
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    candidate_dirs.extend([script_dir.parent.parent, script_dir.parent, script_dir])
    for base in candidate_dirs:
        gguf_py = base / "gguf-py"
        if (gguf_py / "gguf").is_dir():
            sys.path.insert(0, str(gguf_py))
            return


def ggml_type_names(llama_cpp_dir=None) -> Optional[List[str]]:
    """The ggml type names llama-quantize accepts, or None if unknown.

    llama-quantize's parse_ggml_type() walks every ggml type and compares names
    case-insensitively, so the valid set is exactly what the `gguf` module
    enumerates. Reading it from there keeps this in step with upstream instead of
    freezing a hand-written list.
    """
    _add_gguf_to_path(llama_cpp_dir)
    try:
        from gguf import GGMLQuantizationType
    except ImportError:
        return None
    return sorted(t.name.lower() for t in GGMLQuantizationType)


def validate_ggml_types(fail, args, llama_cpp_dir=None) -> None:
    """Reject unknown --output-tensor-type / --token-embedding-type values.

    These accept any ggml type. When the `gguf` module is reachable we can say so
    precisely; otherwise the value is passed through and llama-quantize rejects
    it at argument-parse time, before loading the model.
    """
    valid = ggml_type_names(llama_cpp_dir)
    if valid is None:
        return
    for option, value in (("--output-tensor-type", getattr(args, "output_tensor_type", None)),
                          ("--token-embedding-type", getattr(args, "token_embedding_type", None))):
        if value and value.lower() not in valid:
            fail(f"{option}: unknown ggml type '{value}'. Valid types: {', '.join(valid)}")


# Quantization types whose non-embedding tensors cannot be produced without an
# importance matrix. llama.cpp raises "this quantization requires an imatrix!"
# after loading the model; catching it here avoids that wasted work.
# Source: tensor_requires_imatrix() in llama.cpp src/llama-quant.cpp.
IMATRIX_REQUIRED_TYPES = frozenset({
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "q2_k_s",
})


# GGUF tensor-name fragments for Mixture-of-Experts weights.
#
# llama-quantize compiles each --tensor-type name into a std::regex and applies
# it with std::regex_search (llama.cpp src/llama-quant.cpp:686), i.e. an
# unanchored match, so a bare fragment matches the full tensor name:
# "ffn_gate_exps" matches "blk.1.ffn_gate_exps.weight". These fragments contain
# no regex metacharacters, so they behave as literals.
MOE_EXPERT_TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
MOE_ROUTER_TENSORS = ("ffn_gate_inp",)


def moe_tensor_type_args(expert_type: str, router_type: str) -> List[str]:
    """Build --tensor-type arguments for MoE expert and router weights.

    ``--moe-expert-quantization`` / ``--moe-router-quantization`` used to be
    inert: upstream llama-quantize had no way to target individual tensors. It
    now does, via ``--tensor-type NAME=TYPE`` (repeatable), so these map onto it
    directly. ``same`` means "leave it to the global --type" and emits nothing.
    """
    args: List[str] = []
    for tensor_type, names in ((expert_type, MOE_EXPERT_TENSORS),
                               (router_type, MOE_ROUTER_TENSORS)):
        if tensor_type and tensor_type != "same":
            for name in names:
                args.extend(["--tensor-type", f"{name}={tensor_type}"])
    return args


def read_gguf_metadata(input_file: Path, llama_cpp_dir=None) -> Dict[str, Any]:
    """Read selected key/value metadata from a GGUF file.

    Returns the architecture plus, for Mixture-of-Experts models, the expert
    counts. Expert weights are *stacked* into a single tensor per projection, so
    the number of experts can only be read from metadata - it cannot be counted
    from tensor names.
    """
    _add_gguf_to_path(llama_cpp_dir)
    from gguf import GGUFReader

    reader = GGUFReader(str(input_file))
    fields = {name: field for name, field in reader.fields.items()}

    def value(key):
        field = fields.get(key)
        if field is None:
            return None
        try:
            return field.contents()
        except Exception:
            return None

    arch = value("general.architecture")
    metadata: Dict[str, Any] = {"architecture": arch}
    if arch:
        metadata["expert_count"] = value(f"{arch}.expert_count")
        metadata["expert_used_count"] = value(f"{arch}.expert_used_count")
        metadata["block_count"] = value(f"{arch}.block_count")
    return metadata


def read_gguf_tensors(input_file: Path, llama_cpp_dir=None) -> List[Dict[str, Any]]:
    """Read tensor metadata straight out of a GGUF file.

    Replaces an earlier approach that shelled out to ``llama-quantize --dry-run
    --verbose`` and regex-scraped stdout. That could not work: ``--verbose`` is
    not a llama-quantize option (it exits 1), and the dry-run output reports
    aggregate per-type counts rather than the per-tensor lines the pattern
    expected. Reading the file is also faster, portable, and needs no binary.
    """
    _add_gguf_to_path(llama_cpp_dir)
    try:
        from gguf import GGUFReader
    except ImportError as e:
        raise ImportError(
            "Could not import the gguf module, which is needed to analyse a GGUF "
            "file. Install it with `pip install gguf`, or set LLAMA_CPP_DIR to a "
            "llama.cpp checkout that contains gguf-py, or pass --llama-cpp-dir."
        ) from e

    reader = GGUFReader(str(input_file))
    tensors = []
    for tensor in reader.tensors:
        dims = [int(d) for d in tensor.shape]
        while len(dims) < 4:
            dims.append(1)
        tensors.append({
            "name": str(tensor.name),
            "dimensions": dims[:4],
            "type": str(tensor.tensor_type.name).lower(),
            "size_mb": int(tensor.n_bytes) / (1024 * 1024),
        })
    return tensors


def analyze_model_structure(input_file: Path, verbose: bool = False, llama_cpp_dir=None) -> Dict[str, Any]:
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
    
    try:
        tensor_info = read_gguf_tensors(input_file, llama_cpp_dir)
        # llama-quantize reports these only when it runs; reading the file gives
        # us the true source size instead, and the projected size is not known
        # without invoking the quantizer.
        model_size_mb = sum(t["size_mb"] for t in tensor_info)
        quant_size_mb = None

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
        
        # MoE detection keyed on the tensor names llama.cpp actually emits.
        # An earlier version matched 'gate', 'ffn_up' and 'ffn_down', which appear
        # in *every* dense feed-forward block, so every model looked like an MoE.
        # Stacked expert weights end in `_exps`; the router is `ffn_gate_inp`.
        expert_tensors = []
        router_tensors = []

        for tensor in tensor_info:
            name = tensor["name"].lower()
            if any(fragment in name for fragment in MOE_EXPERT_TENSORS) or "_exps" in name:
                expert_tensors.append(tensor)
            elif any(fragment in name for fragment in MOE_ROUTER_TENSORS) or "exp_probs" in name:
                router_tensors.append(tensor)

        moe_tensors = expert_tensors + router_tensors
        # Experts are the load-bearing signal: a router alone is not an MoE.
        has_moe = len(expert_tensors) > 0
        gate_tensors = router_tensors
        
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
    if args.analyze_model:
        logger.info("Analyzing model structure to detect MoE components...")
        model_analysis = analyze_model_structure(args.model, args.verbose, args.llama_cpp_dir)

        if "error" in model_analysis:
            logger.warning(f"Model analysis failed: {model_analysis['error']}")
            logger.warning("Continuing with standard quantization...")
        else:
            has_moe = model_analysis.get("has_moe", False)
            if has_moe:
                logger.info("Detected Mixture of Experts (MoE) architecture in the model")
                logger.info(f"Found {len(model_analysis.get('moe_tensors', []))} MoE-related tensors")
    
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

    # Importance matrix (and any per-tensor include/exclude scoping)
    cmd.extend(imatrix_args(args))
    
    # Force MoE detection for models with Scout or MoE in their name
    model_name = args.model.name.lower()
    if "scout" in model_name or "moe" in model_name:
        has_moe = True
        logger.info(f"Forcing MoE detection based on model name: {args.model.name}")

    # Selective per-tensor quantization is expressed with llama-quantize's
    # --tensor-type NAME=TYPE, which may be repeated. Expert and router weights
    # are matched by their GGUF tensor-name fragments.
    cmd.extend(moe_tensor_type_args(
        args.moe_expert_quantization, args.moe_router_quantization
    ))

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
    
    # Single-pass quantization using llama-quantize
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

    # --output-tensor-type/--token-embedding-type accept any ggml type, so they
    # are validated here rather than by argparse: the check needs the `gguf`
    # module, which may only be reachable via --llama-cpp-dir. Runs after logging
    # is configured so the message is formatted like every other error.
    def _type_error(message):
        logger.error(message)
        sys.exit(2)

    validate_ggml_types(_type_error, args, args.llama_cpp_dir)
    
    # `auto` is not a quantization type - it selects analysis-only mode, with or
    # without --analyze-model. Previously this branch also required
    # --analyze-model, so `--type auto` on its own fell through and was forwarded
    # to llama-quantize, which rejects it with "invalid ftype 'auto'".
    if args.type == "auto":
        # In this case, we're just analyzing the model without quantizing
        logger.info("Running in analysis-only mode")
        try:
            # Verify the model file exists
            if not args.model.exists():
                logger.error(f"Model file not found: {args.model}")
                return 1
                
            # Analyze the model structure
            analysis_results = analyze_model_structure(
                args.model, verbose=True, llama_cpp_dir=args.llama_cpp_dir)
            
            if "error" in analysis_results:
                logger.error(f"Error analyzing model: {analysis_results['error']}")
                return 1
                
            # Provide quantization recommendations based on analysis
            has_moe = analysis_results.get("has_moe", False)
            if has_moe:
                logger.info("\n===== Quantization Recommendations for MoE Model =====")
                logger.info("This model contains Mixture of Experts (MoE) architecture.")
                logger.info("Recommended quantization settings:")
                logger.info("  1. For better quality: --type q5_k --leave-output-tensor --token-embedding-type f16")
                logger.info("  2. For better compression: --type q4_k_m")
                logger.info("  3. For balanced approach: --type q4_k --output-tensor-type f16")
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
