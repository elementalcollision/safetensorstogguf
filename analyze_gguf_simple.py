#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import struct

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze-gguf-simple")

def get_type_size(tensor_type):
    """Get the size in bytes for a given tensor type"""
    # Mapping of tensor types to their sizes in bytes
    type_sizes = {
        0: 4,   # F32
        1: 2,   # F16
        2: 1,   # Q4_0
        3: 1,   # Q4_1
        7: 1,   # Q8_0
        8: 1,   # Q5_0
        9: 1,   # Q5_1
        10: 0.5, # Q2_K
        11: 0.75, # Q3_K_S
        12: 0.75, # Q3_K_M
        13: 0.75, # Q3_K_L
        14: 1,   # Q4_K_S
        15: 1,   # Q4_K_M
        16: 1.25, # Q5_K_S
        17: 1.25, # Q5_K_M
        18: 1.5,  # Q6_K
        32: 2,   # BF16
    }
    return type_sizes.get(tensor_type, 4)  # Default to F32 if type is unknown

def analyze_gguf_model(model_path):
    """
    Analyze a GGUF model file and print detailed information about its structure.
    
    Args:
        model_path: Path to the GGUF model file
    """
    logger.info(f"Analyzing GGUF model: {model_path}")
    
    # Check if the file exists
    if not os.path.exists(model_path):
        logger.error(f"File not found: {model_path}")
        return
    
    # Get file size
    file_size = os.path.getsize(model_path)
    logger.info(f"File size: {file_size / (1024 * 1024 * 1024):.2f} GB")
    
    # Use llama-quantize to get tensor information
    try:
        import subprocess
        cmd = ["/Users/dave/llama.cpp/build/bin/llama-quantize", "--dry-run", "--verbose", model_path, "/dev/null", "q4_0"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"llama-quantize returned non-zero exit code: {result.returncode}")
            logger.warning(f"Error output: {result.stderr}")
        
        # Parse the output to extract tensor information
        tensor_info = []
        current_tensor = None
        
        for line in result.stdout.split('\n'):
            if '[' in line and ']' in line and '-' in line and 'type' in line and 'size' in line:
                # This line contains tensor information
                parts = line.split('-')
                if len(parts) >= 2:
                    tensor_name = parts[1].strip()
                    shape_str = line.split('[')[1].split(']')[0].strip()
                    shape = [int(x.strip()) for x in shape_str.split(',')]
                    
                    type_str = line.split('type =')[1].split(',')[0].strip()
                    size_str = line.split('size =')[1].strip()
                    size_mb = float(size_str.split(' ')[0])
                    
                    tensor_info.append({
                        'name': tensor_name,
                        'shape': shape,
                        'type': type_str,
                        'size_mb': size_mb
                    })
        
        # Analyze tensor distribution
        logger.info(f"Total number of tensors: {len(tensor_info)}")
        
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
            logger.info(f"  {tensor_type}: {count} tensors ({count/len(tensor_info)*100:.2f}%), {size_mb:.2f} MB ({size_mb/total_size_mb*100:.2f}%)")
        
        # Print total model size
        logger.info(f"\nTotal model size from tensors: {total_size_mb / 1024:.2f} GB")
        
        # Find and print the 10 largest tensors
        largest_tensors = sorted(tensor_info, key=lambda x: x['size_mb'], reverse=True)[:10]
        logger.info(f"\nTop 10 largest tensors:")
        for tensor in largest_tensors:
            logger.info(f"  {tensor['name']} - Shape: {tensor['shape']}, Type: {tensor['type']}, Size: {tensor['size_mb']:.2f} MB")
        
        # Look for MoE-specific tensors
        moe_keywords = ['expert', 'router', 'moe', 'gate', 'ffn_gate', 'ffn_up', 'ffn_down']
        moe_tensors = [t for t in tensor_info if any(kw in t['name'].lower() for kw in moe_keywords)]
        
        if moe_tensors:
            logger.info(f"\nFound {len(moe_tensors)} potential MoE-related tensors:")
            moe_size_mb = sum(t['size_mb'] for t in moe_tensors)
            logger.info(f"Total MoE tensor size: {moe_size_mb:.2f} MB ({moe_size_mb/total_size_mb*100:.2f}% of model)")
            
            # Group by tensor type
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
            
            # Show the first 20 MoE tensors
            logger.info(f"\nSample of MoE tensors:")
            for tensor in moe_tensors[:20]:
                logger.info(f"  {tensor['name']} - Shape: {tensor['shape']}, Type: {tensor['type']}, Size: {tensor['size_mb']:.2f} MB")
            
            # Analyze expert patterns
            expert_tensors = [t for t in moe_tensors if 'expert' in t['name'].lower()]
            if expert_tensors:
                logger.info(f"\nFound {len(expert_tensors)} expert tensors")
                
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
                    logger.info(f"Detected approximately {num_experts} experts in the model")
                    
                    # Show distribution of expert tensors
                    logger.info(f"Expert tensor distribution:")
                    for expert_num, count in sorted(expert_patterns.items(), key=lambda x: int(x[0])):
                        logger.info(f"  Expert {expert_num}: {count} tensors")
        else:
            logger.info("\nNo MoE-specific tensors found in the model.")
        
        # Analyze tensor name patterns to identify model structure
        tensor_patterns = {}
        for tensor in tensor_info:
            name = tensor['name']
            # Create a pattern by replacing numbers with #
            pattern = ''.join(['#' if c.isdigit() else c for c in name])
            
            if pattern not in tensor_patterns:
                tensor_patterns[pattern] = {'count': 0, 'examples': []}
            
            tensor_patterns[pattern]['count'] += 1
            if len(tensor_patterns[pattern]['examples']) < 3:
                tensor_patterns[pattern]['examples'].append(name)
        
        # Print the most common patterns
        common_patterns = sorted(tensor_patterns.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        logger.info(f"\nTop 10 tensor name patterns:")
        for pattern, info in common_patterns:
            logger.info(f"  {pattern}: {info['count']} tensors")
            logger.info(f"    Examples: {', '.join(info['examples'])}")
        
    except Exception as e:
        logger.error(f"Failed to analyze model: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Analyze GGUF model structure")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model file")
    args = parser.parse_args()
    
    analyze_gguf_model(args.model)

if __name__ == "__main__":
    main()
