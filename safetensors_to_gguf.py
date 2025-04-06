#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safetensors_to_gguf.py - A CLI tool to convert safetensors files to GGUF format

This tool leverages llama.cpp's conversion utilities to convert safetensors model files
to GGUF format for use with llama.cpp inference.
"""

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

# Global variables that will be set in setup_llama_cpp_path
LLAMA_CPP_PATH = None
Model = None
LlamaModel = None
Llama4Model = None

# Configure logging
logger = logging.getLogger("safetensors-to-gguf")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def setup_llama_cpp_path(llama_cpp_dir=None):
    """Set up the llama.cpp path and import necessary modules"""
    global LLAMA_CPP_PATH, Model, LlamaModel, Llama4Model
    
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
            convert_script = path / "convert_hf_to_gguf.py"
            if convert_script.exists():
                llama_cpp_dir = path
                break
    
    if llama_cpp_dir is None or not (llama_cpp_dir / "convert_hf_to_gguf.py").exists():
        raise ValueError(
            "Could not find llama.cpp directory. Please specify it using --llama-cpp-dir. "
            "The directory should contain convert_hf_to_gguf.py."
        )
    
    # Set the global path
    LLAMA_CPP_PATH = llama_cpp_dir
    
    # Add to Python path
    sys.path.insert(0, str(LLAMA_CPP_PATH))
    sys.path.insert(1, str(LLAMA_CPP_PATH / 'gguf-py'))
    
    # Import necessary modules
    try:
        import gguf
    except ImportError:
        raise ImportError("Could not import gguf module. Make sure llama.cpp is properly installed.")
    
    try:
        # Use importlib to import the module without executing its main function
        import importlib.util
        convert_script_path = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"
        
        if not convert_script_path.exists():
            raise FileNotFoundError(f"Could not find {convert_script_path}")
            
        spec = importlib.util.spec_from_file_location(
            "convert_hf_to_gguf", 
            str(convert_script_path)
        )
        convert_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(convert_module)
        
        # Get the Model class and related functions
        global Model, LlamaModel, Llama4Model
        Model = convert_module.Model
        LlamaModel = convert_module.LlamaModel
        
        # Now register the Llama4Model class
        @Model.register("Llama4ForConditionalGeneration")
        class Llama4ModelImpl(LlamaModel):
            """Llama4 model implementation"""
            # Use the same model architecture as LlamaModel
            model_arch = LlamaModel.model_arch
            
            def find_hparam(self, keys, optional=False):
                """Override to handle nested configuration in Llama4 models"""
                # First try the standard approach
                try:
                    return super().find_hparam(keys, optional=True)
                except ValueError:
                    # If not found, try looking in text_config
                    if "text_config" in self.hparams:
                        for key in keys:
                            if key in self.hparams["text_config"]:
                                return self.hparams["text_config"][key]
                    
                    # If still not found and not optional, raise error
                    if not optional:
                        raise ValueError(f"could not find any of: {keys}")
                    return None
            
            def modify_tensors(self, data_torch, name, bid=None):
                """Override to handle Llama4 model tensors"""
                logger = logging.getLogger("safetensors-to-gguf")
                
                # Skip multimodal tensors
                if "multi_modal_projector" in name or "vision_model" in name:
                    logger.debug(f"Skipping multimodal tensor: {name}")
                    return []
                
                # Skip MoE-specific tensors (router weights and expert layers)
                if "feed_forward.router" in name or "feed_forward.experts" in name:
                    logger.debug(f"Skipping MoE tensor: {name}")
                    return []
                
                # For all other tensors, use the standard handling
                return super().modify_tensors(data_torch, name, bid)
            
            def set_gguf_parameters(self):
                """Override to handle Llama-4 specific parameters"""
                logger = logging.getLogger("safetensors-to-gguf")
                
                # Make sure vocab_size is set correctly
                if "vocab_size" not in self.hparams and "text_config" in self.hparams and "vocab_size" in self.hparams["text_config"]:
                    self.hparams["vocab_size"] = self.hparams["text_config"]["vocab_size"]
                
                # Call the parent method to set standard parameters
                super().set_gguf_parameters()
                
                # Add context length parameter
                context_length = None
                
                # Try to get context length from max_position_embeddings in text_config
                if "text_config" in self.hparams and "max_position_embeddings" in self.hparams["text_config"]:
                    context_length = self.hparams["text_config"]["max_position_embeddings"]
                    logger.info(f"Using max_position_embeddings from text_config as context_length: {context_length}")
                # Fallback to max_position_embeddings in root config
                elif "max_position_embeddings" in self.hparams:
                    context_length = self.hparams["max_position_embeddings"]
                    logger.info(f"Using max_position_embeddings as context_length: {context_length}")
                # Fallback to a default value for Llama-4
                else:
                    context_length = 8192  # Default context length for Llama-4
                    logger.warning(f"Could not find context length in model config, using default: {context_length}")
                
                # Add the context length parameter to the GGUF file
                self.gguf_writer.add_u32("llama.context_length", context_length)
                logger.info(f"Added context_length parameter: {context_length}")
            
            def set_vocab(self):
                """Override to handle Llama-4 tokenizer"""
                logger = logging.getLogger("safetensors-to-gguf")
                logger.info("Using Llama-4 custom tokenizer handling")
                
                # Check if tokenizer.json exists instead of tokenizer.model
                tokenizer_json_path = os.path.join(self.dir_model, "tokenizer.json")
                if os.path.exists(tokenizer_json_path):
                    logger.info(f"Found tokenizer.json at {tokenizer_json_path}")
                    self._set_vocab_from_tokenizer_json(tokenizer_json_path)
                else:
                    # Fall back to standard method
                    logger.warning("No tokenizer.json found, falling back to standard method")
                    super().set_vocab()
            
            def _set_vocab_from_tokenizer_json(self, tokenizer_path):
                """Set vocabulary from tokenizer.json file"""
                logger = logging.getLogger("safetensors-to-gguf")
                logger.info(f"Loading vocabulary from {tokenizer_path}")
                
                import json
                with open(tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                
                # Extract vocabulary from tokenizer.json
                if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                    vocab = tokenizer_data['model']['vocab']
                    logger.info(f"Loaded vocabulary with {len(vocab)} tokens")
                    
                    # Create token list
                    tokens = []
                    scores = []
                    toktypes = []
                    
                    # Process each token
                    for token, token_id in vocab.items():
                        # Convert token to bytes
                        token_bytes = token.encode('utf-8')
                        tokens.append(token_bytes)
                        
                        # Use default score
                        scores.append(0.0)
                        
                        # Use default token type (normal)
                        toktypes.append(0)
                    
                    # Add tokens to GGUF
                    self.gguf_writer.add_tokenizer_model("llama")
                    self.gguf_writer.add_token_list(tokens)
                    self.gguf_writer.add_token_scores(scores)
                    
                    # Add token types if supported
                    if hasattr(self.gguf_writer, 'add_token_types'):
                        self.gguf_writer.add_token_types(toktypes)
                    
                    # Handle special tokens
                    # Get special token IDs from config
                    special_tokens = {}
                    
                    # Check text_config for special token IDs
                    if 'text_config' in self.hparams:
                        if 'bos_token_id' in self.hparams['text_config']:
                            special_tokens['bos_token_id'] = self.hparams['text_config']['bos_token_id']
                        
                        if 'eos_token_id' in self.hparams['text_config']:
                            eos_tokens = self.hparams['text_config']['eos_token_id']
                            # Handle both single value and list
                            if isinstance(eos_tokens, list):
                                special_tokens['eos_token_id'] = eos_tokens[0]  # Use first one
                            else:
                                special_tokens['eos_token_id'] = eos_tokens
                    
                    # Add special tokens to GGUF
                    if 'bos_token_id' in special_tokens:
                        logger.info(f"Setting BOS token ID: {special_tokens['bos_token_id']}")
                        self.gguf_writer.add_bos_token_id(special_tokens['bos_token_id'])
                    
                    if 'eos_token_id' in special_tokens:
                        logger.info(f"Setting EOS token ID: {special_tokens['eos_token_id']}")
                        self.gguf_writer.add_eos_token_id(special_tokens['eos_token_id'])
                else:
                    raise ValueError("Could not find vocabulary in tokenizer.json")
            
            def map_tensor_name(self, name, try_suffixes=(".weight", ".bias")):
                """Override to handle Llama-4 tensor naming structure"""
                # First try the standard mapping
                try:
                    return super().map_tensor_name(name, try_suffixes)
                except ValueError:
                    # Handle Llama-4 specific tensor naming
                    logger = logging.getLogger("safetensors-to-gguf")
                    logger.debug(f"Attempting to map Llama-4 tensor: {name}")
                    
                    # Handle shared expert tensors
                    if "feed_forward.shared_expert" in name:
                        # Map feed-forward layers
                        if "gate_proj" in name:
                            return "feed_forward.gate_proj"
                        elif "up_proj" in name:
                            return "feed_forward.up_proj"
                        elif "down_proj" in name:
                            return "feed_forward.down_proj"
                    
                    # Handle other Llama-4 specific tensor names
                    if "language_model.model.embed_tokens" in name:
                        return "token_embd"
                    elif "language_model.model.norm" in name:
                        return "output_norm"
                    elif "language_model.lm_head" in name:
                        return "output"
                    
                    # Handle attention layers
                    for layer_type in ["input_layernorm", "post_attention_layernorm", "self_attn.q_proj", 
                                    "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
                        if layer_type in name:
                            # Extract the layer number
                            parts = name.split(".")
                            for part in parts:
                                if part.startswith("layers"):
                                    layer_parts = part.split("layers.")
                                    if len(layer_parts) > 1 and layer_parts[1].isdigit():
                                        layer_num = int(layer_parts[1])
                                        # Map to standard llama.cpp tensor names
                                        if "input_layernorm" in layer_type:
                                            return f"blk.{layer_num}.attn_norm"
                                        elif "post_attention_layernorm" in layer_type:
                                            return f"blk.{layer_num}.ffn_norm"
                                        elif "self_attn.q_proj" in layer_type:
                                            return f"blk.{layer_num}.attn_q"
                                        elif "self_attn.k_proj" in layer_type:
                                            return f"blk.{layer_num}.attn_k"
                                        elif "self_attn.v_proj" in layer_type:
                                            return f"blk.{layer_num}.attn_v"
                                        elif "self_attn.o_proj" in layer_type:
                                            return f"blk.{layer_num}.attn_output"
                    
                    # If we get here, we couldn't map the tensor name
                    logger.warning(f"Could not map tensor: {name}")
                    raise ValueError(f"Can not map tensor {name!r}")
        
        # Replace the global Llama4Model with the implementation
        Llama4Model = Llama4ModelImpl
        
        return LLAMA_CPP_PATH
    except Exception as e:
        raise ImportError(f"Error importing convert_hf_to_gguf.py: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert safetensors model files to GGUF format for use with llama.cpp"
    )
    
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to the directory containing the model's safetensors files"
    )
    
    parser.add_argument(
        "--outfile", type=Path,
        help="Path to write the output GGUF file (default: model directory name with .gguf extension)"
    )
    
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="auto",
        help="Output data type (default: auto, which tries to detect from the model)"
    )
    
    parser.add_argument(
        "--bigendian", action="store_true",
        help="Use big endian format for output file (default: little endian / x86)"
    )
    
    parser.add_argument(
        "--vocab-only", action="store_true",
        help="Extract only the vocabulary"
    )
    
    parser.add_argument(
        "--model-name", type=str,
        help="Override the model name in the GGUF file metadata"
    )
    
    parser.add_argument(
        "--metadata", type=Path,
        help="Path to a JSON file containing metadata to add to the GGUF file"
    )
    
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads to use for conversion (default: number of CPU cores)"
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

def verify_safetensors_model(model_dir: Path) -> bool:
    """
    Verify that the model directory contains safetensors files.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        bool: True if safetensors files are found, False otherwise
    """
    if not model_dir.is_dir():
        logger.error(f"Error: {model_dir} is not a directory")
        return False
    
    # Check for safetensors files
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        logger.error(f"Error: No safetensors files found in {model_dir}")
        return False
    
    logger.info(f"Found {len(safetensors_files)} safetensors files in {model_dir}")
    return True

def convert_safetensors_to_gguf(args):
    """
    Convert safetensors model to GGUF format.
    
    Args:
        args: Command line arguments
    """
    # Verify that the model directory contains safetensors files
    if not verify_safetensors_model(args.model):
        sys.exit(1)
    
    # Set up threading if specified
    if args.threads is not None:
        torch_threads = args.threads
        logger.info(f"Setting torch threads to {torch_threads}")
        import torch
        torch.set_num_threads(torch_threads)
        
        # Set threading parameters for thread safety
        threading.current_thread().name = "MainThread"
    
    # Map output type to GGUF file type
    import gguf
    ftype_map = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }
    output_type = ftype_map[args.outtype]
    
    # Load model hyperparameters
    logger.info(f"Loading model: {args.model.name}")
    hparams = Model.load_hparams(args.model)
    
    # Debug: Print the hyperparameters structure
    if args.verbose:
        logger.debug(f"Model hyperparameters: {json.dumps(hparams, indent=2, default=str)}")
        if "text_config" in hparams:
            logger.debug(f"text_config: {json.dumps(hparams['text_config'], indent=2, default=str)}")
            logger.debug(f"num_hidden_layers: {hparams['text_config'].get('num_hidden_layers')}")
    
    try:
        # Get model architecture
        model_architecture = hparams["architectures"][0]
        logger.info(f"Model architecture: {model_architecture}")
        
        # For Llama4 models, ensure we have the necessary parameters
        if model_architecture == "Llama4ForConditionalGeneration":
            # Pre-process the hparams to make them compatible with the converter
            if "text_config" in hparams and "num_hidden_layers" in hparams["text_config"]:
                # Copy essential parameters to the top level
                hparams["num_hidden_layers"] = hparams["text_config"]["num_hidden_layers"]
                logger.info(f"Using num_hidden_layers from text_config: {hparams['num_hidden_layers']}")
                
                # Copy other essential parameters if needed
                for param in ["hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size"]:
                    if param in hparams["text_config"]:
                        hparams[param] = hparams["text_config"][param]
                        logger.debug(f"Copied {param}: {hparams[param]}")
        
        # Get model class
        try:
            model_class = Model.from_model_architecture(model_architecture)
            logger.info(f"Using model class: {model_class.__name__}")
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)
        
        # Create model instance
        import torch
        with torch.inference_mode():
            logger.info("Creating model instance...")
            model_instance = model_class(
                args.model, 
                output_type, 
                args.outfile,
                is_big_endian=args.bigendian, 
                use_temp_file=False,
                eager=False,
                metadata_override=args.metadata, 
                model_name=args.model_name,
                hparams=hparams  # Pass our modified hparams
            )
            
            # Export model
            if args.vocab_only:
                logger.info("Exporting model vocabulary...")
                model_instance.write_vocab()
                logger.info(f"Model vocabulary successfully exported to {model_instance.fname_out}")
            else:
                logger.info("Exporting model...")
                model_instance.write()
                logger.info(f"Model successfully exported to {model_instance.fname_out}")
    
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("safetensors-to-gguf")
    
    # Set up llama.cpp path
    try:
        llama_cpp_dir = setup_llama_cpp_path(args.llama_cpp_dir)
        logger.info(f"Using llama.cpp directory: {llama_cpp_dir}")
    except Exception as e:
        logger.error(f"Error setting up llama.cpp path: {e}")
        return 1
    
    # Convert the model
    try:
        convert_safetensors_to_gguf(args)
        return 0
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
