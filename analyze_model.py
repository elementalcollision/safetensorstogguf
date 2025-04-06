#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import subprocess
import re
from pathlib import Path
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
    
    # Find llama-quantize binary
    if llama_cpp_dir is None:
        # Try to find it relative to the script location
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            script_dir.parent.parent,  # If script is in llama.cpp/some_dir/safetensors-to-gguf
            script_dir.parent,         # If script is in llama.cpp/safetensors-to-gguf
            Path("/Users/dave/llama.cpp")  # Direct path to llama.cpp
        ]
        
        for path in possible_paths:
            quantize_binary = path / "build" / "bin" / "llama-quantize"
            if quantize_binary.exists():
                llama_cpp_dir = path
                logger.info(f"Found llama.cpp directory at: {llama_cpp_dir}")
                break
    
    if llama_cpp_dir is None:
        logger.error("Could not find llama.cpp directory with llama-quantize binary")
        return None
    
    quantize_binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_binary.exists():
        logger.error(f"llama-quantize binary not found at {quantize_binary}")
        return None
    
    # Run llama-quantize with --dry-run to analyze the model
    cmd = [str(quantize_binary), "--dry-run", "--verbose", str(model_path), "/dev/null", "q4_0"]
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"llama-quantize returned non-zero exit code: {result.returncode}")
            logger.warning(f"Error output: {result.stderr}")
            
            # Check if the model is already quantized
            if "already quantized" in result.stderr:
                logger.info("Model appears to be already quantized")
                return {"is_quantized": True, "error": result.stderr}
            
            # Try to extract useful information from stderr
            return {"error": result.stderr}
    except Exception as e:
        logger.error(f"Error running llama-quantize: {e}")
        return {"error": str(e)}
    
    # Parse the output to extract tensor information
    tensor_info = []
    
    # Regular expression to match tensor information lines
    tensor_pattern = r'\[\s*(\d+)/\s*(\d+)\]\s+(.*?)\s+-\s+\[(.*?)\],\s+type\s+=\s+(\S+),\s+size\s+=\s+([\d.]+)\s+MB'
    
    for line in result.stdout.split('\n'):
        match = re.search(tensor_pattern, line)
        if match:
            tensor_idx = int(match.group(1))
            total_tensors = int(match.group(2))
            tensor_name = match.group(3).strip()
            shape_str = match.group(4).strip()
            shape = [int(x.strip()) for x in shape_str.split(',')]
            tensor_type = match.group(5).strip()
            size_mb = float(match.group(6))
            
            tensor_info.append({
                'index': tensor_idx,
                'name': tensor_name,
                'shape': shape,
                'type': tensor_type,
                'size_mb': size_mb
            })
    
    # If we couldn't extract tensor information, try a different approach
    if not tensor_info:
        logger.warning("Could not extract tensor information from llama-quantize output")
        
        # Check if the model name contains "Scout" to force MoE detection
        model_name = os.path.basename(model_path)
        is_moe = "scout" in model_name.lower() or "moe" in model_name.lower()
        
        if is_moe:
            logger.info(f"Forcing MoE detection based on model name: {model_name}")
            return {
                "is_moe": True,
                "forced_detection": True,
                "model_name": model_name,
                "error": "Could not extract tensor information, but model name suggests MoE architecture"
            }
        
        return {"error": "Could not extract tensor information from llama-quantize output"}
    
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
    
    # Check for MoE components
    moe_keywords = ['expert', 'router', 'moe', 'gate', 'ffn_gate', 'ffn_up', 'ffn_down']
    moe_tensors = [t for t in tensor_info if any(kw in t['name'].lower() for kw in moe_keywords)]
    
    is_moe = len(moe_tensors) > 0
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
        
        if 'expert' in name:
            categories['expert'].append(tensor)
        elif 'router' in name or 'gate' in name:
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
        
        # Try to determine the number of experts
        expert_patterns = {}
        for tensor in expert_tensors:
            name = tensor['name']
            # Extract expert number using a simple heuristic
            parts = name.split('expert')
            if len(parts) > 1:
                expert_num = ''.join([c for c in parts[1] if c.isdigit()])
                if expert_num:
                    if expert_num not in expert_patterns:
                        expert_patterns[expert_num] = 0
                    expert_patterns[expert_num] += 1
        
        if expert_patterns:
            num_experts = len(expert_patterns)
            logger.info(f"\nDetected {num_experts} experts in the model")
            
            # Show distribution of expert tensors
            logger.info(f"Expert tensor distribution:")
            for expert_num, count in sorted(expert_patterns.items(), key=lambda x: int(x[0])):
                logger.info(f"  Expert {expert_num}: {count} tensors")
    
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
