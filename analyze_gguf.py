#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze-gguf")

# Add the llama.cpp/gguf directory to the Python path
sys.path.append(str(Path("/Users/dave/llama.cpp")))
sys.path.append(str(Path("/Users/dave/llama.cpp/gguf")))

try:
    import gguf
except ImportError:
    logger.error("Could not import gguf module. Make sure the llama.cpp/gguf directory is in your Python path.")
    sys.exit(1)

def analyze_gguf_model(model_path):
    """
    Analyze a GGUF model file and print detailed information about its structure.
    
    Args:
        model_path: Path to the GGUF model file
    """
    logger.info(f"Analyzing GGUF model: {model_path}")
    
    # Load the GGUF model
    try:
        gguf_reader = gguf.GGUFReader(model_path)
        logger.info(f"Successfully loaded GGUF model")
    except Exception as e:
        logger.error(f"Failed to load GGUF model: {e}")
        return
    
    # Get model metadata
    try:
        metadata = {}
        for k in gguf_reader.keys():
            metadata[k] = gguf_reader.get_val(k)
        
        logger.info(f"Model metadata:")
        for k, v in metadata.items():
            logger.info(f"  {k}: {v}")
    except Exception as e:
        logger.error(f"Failed to get model metadata: {e}")
    
    # Analyze tensor distribution
    try:
        tensor_count = len(gguf_reader.tensors)
        logger.info(f"Total number of tensors: {tensor_count}")
        
        tensor_types = {}
        tensor_sizes = {}
        tensor_shapes = {}
        total_size = 0
        
        # Collect tensor statistics
        for i, tensor in enumerate(gguf_reader.tensors):
            tensor_name = tensor.name
            tensor_type = tensor.tensor_type
            tensor_shape = tensor.shape
            tensor_size = np.prod(tensor_shape) * gguf.get_type_size(tensor_type)
            total_size += tensor_size
            
            # Update tensor type statistics
            if tensor_type not in tensor_types:
                tensor_types[tensor_type] = 0
            tensor_types[tensor_type] += 1
            
            # Update tensor size statistics
            size_mb = tensor_size / (1024 * 1024)
            tensor_sizes[tensor_name] = size_mb
            tensor_shapes[tensor_name] = tensor_shape
            
            # Print information for the first 10 tensors and the 10 largest tensors
            if i < 10:
                logger.info(f"Tensor {i+1}: {tensor_name} - Shape: {tensor_shape}, Type: {gguf.type_name(tensor_type)}, Size: {size_mb:.2f} MB")
        
        # Print tensor type statistics
        logger.info(f"\nTensor type distribution:")
        for tensor_type, count in tensor_types.items():
            logger.info(f"  {gguf.type_name(tensor_type)}: {count} tensors ({count/tensor_count*100:.2f}%)")
        
        # Print total model size
        logger.info(f"\nTotal model size: {total_size / (1024 * 1024 * 1024):.2f} GB")
        
        # Find and print the 10 largest tensors
        largest_tensors = sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"\nTop 10 largest tensors:")
        for tensor_name, size_mb in largest_tensors:
            logger.info(f"  {tensor_name} - Shape: {tensor_shapes[tensor_name]}, Size: {size_mb:.2f} MB")
        
        # Look for MoE-specific tensors
        moe_tensors = [t for t in tensor_sizes.keys() if any(kw in t for kw in ['expert', 'router', 'moe', 'gate'])]
        logger.info(f"\nFound {len(moe_tensors)} potential MoE-related tensors:")
        for tensor_name in moe_tensors[:20]:  # Show only the first 20 to avoid overwhelming output
            logger.info(f"  {tensor_name} - Shape: {tensor_shapes[tensor_name]}, Size: {tensor_sizes[tensor_name]:.2f} MB")
        
        # Analyze tensor name patterns
        logger.info("\nAnalyzing tensor name patterns:")
        tensor_patterns = {}
        for tensor_name in tensor_sizes.keys():
            # Extract the pattern by replacing numbers with #
            pattern = ''.join(['#' if c.isdigit() else c for c in tensor_name])
            if pattern not in tensor_patterns:
                tensor_patterns[pattern] = 0
            tensor_patterns[pattern] += 1
        
        # Print the most common patterns
        common_patterns = sorted(tensor_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top 10 tensor name patterns:")
        for pattern, count in common_patterns:
            logger.info(f"  {pattern}: {count} tensors")
            
    except Exception as e:
        logger.error(f"Failed to analyze tensor distribution: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Analyze GGUF model structure")
    parser.add_argument("--model", type=str, required=True, help="Path to the GGUF model file")
    args = parser.parse_args()
    
    analyze_gguf_model(args.model)

if __name__ == "__main__":
    main()
