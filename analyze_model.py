#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path

# Reuse the quantizer's GGUF readers and MoE tensor-name fragments so the two
# analysis paths cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_gguf import (
    MOE_EXPERT_TENSORS,
    MOE_ROUTER_TENSORS,
    read_gguf_metadata,
    read_gguf_tensors,
)
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze-model")

def analyze_model_structure(model_path, llama_cpp_dir=None):
    """
    Analyze the structure of a GGUF model to identify tensor distribution and MoE components.
    
    Args:
        model_path: Path to the GGUF model file
        llama_cpp_dir: Path to the llama.cpp directory
        
    Returns:
        A dictionary containing analysis results
    """
    logger.info(f"Analyzing model structure: {model_path}")
    
    # No llama-quantize binary is needed any more - the analysis reads the GGUF
    # directly. --llama-cpp-dir is still honoured, but only so the bundled
    # gguf-py can be located when the `gguf` package is not installed. It is
    # passed through explicitly rather than exported to the environment, so the
    # flag takes precedence over any pre-existing LLAMA_CPP_DIR.

    # Read the model directly instead of shelling out to llama-quantize.
    #
    # The previous implementation ran `llama-quantize --dry-run --verbose <model>
    # /dev/null q4_0` and regex-scraped the result, which could not work:
    # --verbose is not a llama-quantize option (it prints usage and exits 1),
    # llama.cpp logs through LLAMA_LOG_INFO whose default sink is stderr rather
    # than the stdout being read, and sizes are reported in MiB rather than the
    # MB the pattern required. Reading the GGUF needs no binary at all.
    try:
        tensor_info = [
            {
                "index": i,
                "name": t["name"],
                "shape": [d for d in t["dimensions"] if d > 1] or [1],
                "type": t["type"],
                "size_mb": t["size_mb"],
            }
            for i, t in enumerate(read_gguf_tensors(Path(model_path), llama_cpp_dir))
        ]
        metadata = read_gguf_metadata(Path(model_path), llama_cpp_dir)
    except Exception as e:
        logger.error(f"Could not read GGUF file: {e}")
        return {"error": str(e)}

    if not tensor_info:
        return {"error": f"No tensors found in {model_path}"}


    # Analyze tensor distribution
    total_tensors = len(tensor_info)
    logger.info(f"Total number of tensors: {total_tensors}")
    
    # Collect tensor type statistics
    tensor_types = {}
    total_size_mb = 0
    
    for tensor in tensor_info:
        tensor_type = tensor['type']
        tensor_size_mb = tensor['size_mb']
        total_size_mb += tensor_size_mb
        
        if tensor_type not in tensor_types:
            tensor_types[tensor_type] = {'count': 0, 'size_mb': 0}
        
        tensor_types[tensor_type]['count'] += 1
        tensor_types[tensor_type]['size_mb'] += tensor_size_mb
    
    # Print tensor type statistics
    logger.info(f"\nTensor type distribution:")
    for tensor_type, stats in tensor_types.items():
        count = stats['count']
        size_mb = stats['size_mb']
        logger.info(f"  {tensor_type}: {count} tensors ({count/total_tensors*100:.2f}%), {size_mb:.2f} MB ({size_mb/total_size_mb*100:.2f}%)")
    
    # MoE detection keyed on the tensor names llama.cpp emits. The earlier
    # keyword list included 'gate', 'ffn_up' and 'ffn_down', which appear in
    # every dense feed-forward block, so every model looked like an MoE.
    def _is_expert(name):
        return any(f in name for f in MOE_EXPERT_TENSORS) or "_exps" in name

    def _is_router(name):
        return any(f in name for f in MOE_ROUTER_TENSORS) or "exp_probs" in name

    moe_tensors = [t for t in tensor_info
                   if _is_expert(t['name'].lower()) or _is_router(t['name'].lower())]

    # Experts are the load-bearing signal: a router alone is not an MoE.
    is_moe = any(_is_expert(t['name'].lower()) for t in tensor_info)
    logger.info(f"\nIs MoE model: {is_moe}")
    
    if is_moe:
        logger.info(f"Found {len(moe_tensors)} potential MoE-related tensors")
        moe_size_mb = sum(t['size_mb'] for t in moe_tensors)
        logger.info(f"Total MoE tensor size: {moe_size_mb:.2f} MB ({moe_size_mb/total_size_mb*100:.2f}% of model)")
        
        # Group MoE tensors by type
        moe_types = {}
        for tensor in moe_tensors:
            tensor_type = tensor['type']
            if tensor_type not in moe_types:
                moe_types[tensor_type] = {'count': 0, 'size_mb': 0}
            
            moe_types[tensor_type]['count'] += 1
            moe_types[tensor_type]['size_mb'] += tensor['size_mb']
        
        logger.info(f"\nMoE tensor type distribution:")
        for tensor_type, stats in moe_types.items():
            count = stats['count']
            size_mb = stats['size_mb']
            logger.info(f"  {tensor_type}: {count} tensors ({count/len(moe_tensors)*100:.2f}%), {size_mb:.2f} MB ({size_mb/moe_size_mb*100:.2f}%)")
    
    # Categorize tensors
    categories = {
        'attention': [],
        'ffn': [],
        'expert': [],
        'router': [],
        'embedding': [],
        'norm': [],
        'output': [],
        'other': []
    }
    
    for tensor in tensor_info:
        name = tensor['name'].lower()
        
        if _is_expert(name):
            categories['expert'].append(tensor)
        elif _is_router(name):
            categories['router'].append(tensor)
        elif 'attn' in name:
            categories['attention'].append(tensor)
        elif 'ffn' in name and 'expert' not in name:
            categories['ffn'].append(tensor)
        elif 'embed' in name:
            categories['embedding'].append(tensor)
        elif 'norm' in name:
            categories['norm'].append(tensor)
        elif 'output' in name:
            categories['output'].append(tensor)
        else:
            categories['other'].append(tensor)
    
    # Print category statistics
    logger.info(f"\nTensor categories:")
    for category, tensors in categories.items():
        if tensors:
            category_size_mb = sum(t['size_mb'] for t in tensors)
            logger.info(f"  {category}: {len(tensors)} tensors, {category_size_mb:.2f} MB ({category_size_mb/total_size_mb*100:.2f}%)")
    
    # Analyze expert patterns if this is an MoE model
    num_experts = 0
    expert_size = 0
    
    if is_moe and categories['expert']:
        expert_tensors = categories['expert']
        expert_size = sum(t['size_mb'] for t in expert_tensors)
        
        # The expert count comes from metadata, not from tensor names: llama.cpp
        # stacks all experts of a projection into one tensor (blk.N.ffn_*_exps),
        # so there is no per-expert name to count. The previous heuristic split
        # names on 'expert' and always found nothing.
        num_experts = metadata.get("expert_count") or 0
        used = metadata.get("expert_used_count")
        if num_experts:
            logger.info(f"\n{num_experts} experts in the model"
                        + (f", {used} active per token" if used else ""))
            logger.info(f"Stacked expert tensors: {len(expert_tensors)}"
                        f" ({expert_size:.2f} MB)")
        else:
            logger.info("\nExpert tensors present but no expert_count in metadata")
    
    # Prepare results
    results = {
        "model_path": str(model_path),
        "total_tensors": total_tensors,
        "total_size_mb": total_size_mb,
        "tensor_types": tensor_types,
        "is_moe": is_moe,
        "categories": {k: len(v) for k, v in categories.items()},
        "category_sizes": {k: sum(t['size_mb'] for t in v) for k, v in categories.items() if v},
        "num_experts": num_experts,
        "expert_size_mb": expert_size
    }
    
    if is_moe:
        results["moe_tensors"] = len(moe_tensors)
        results["moe_size_mb"] = moe_size_mb
        results["moe_types"] = moe_types
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze GGUF model structure")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model file")
    parser.add_argument("--llama-cpp-dir", type=str, help="Path to the llama.cpp directory")
    parser.add_argument("--output", type=str, help="Path to write analysis results as JSON")
    args = parser.parse_args()
    
    # Convert string path to Path object if provided
    llama_cpp_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else None
    
    results = analyze_model_structure(args.model, llama_cpp_dir)
    
    if results:
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Analysis results written to {args.output}")
        else:
            logger.info("\nAnalysis results summary:")
            for key, value in results.items():
                if not isinstance(value, dict):
                    logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()
