#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safetensors_to_gguf.py - A CLI tool to convert safetensors files to GGUF format

This tool leverages llama.cpp's conversion utilities to convert safetensors model files
to GGUF format for use with llama.cpp inference.
"""

import argparse
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

# Global variables that will be set in setup_llama_cpp_path
LLAMA_CPP_PATH = None
Model = None        # ModelBase (or, on very old checkouts, Model)
UPSTREAM = None     # UpstreamConverter describing what this llama.cpp provides

# Logging is configured via logging.basicConfig() in main(); fetch the named
# logger here without attaching a second handler (which would double every line).
logger = logging.getLogger("safetensors-to-gguf")


class UpstreamConverter:
    """A version-agnostic view of llama.cpp's conversion machinery.

    llama.cpp's converter has changed shape several times, and pinning any one
    of those shapes is what has repeatedly broken this tool:

    * ``<= 0f5ccd6f`` (2025-08-01) — ``load_hparams(dir_model)``.
    * ``a3a7874``  (2025-08-11, llama.cpp#14737) — ``load_hparams`` gained a
      second *required* argument, ``is_mistral_format``.
    * ``cc7200bf``  (2026-05-15, llama.cpp#17114) — the monolithic
      ``convert_hf_to_gguf.py`` was split into a ``conversion`` package. It now
      re-exports only a handful of helpers and **no model classes**, and the
      class registry is populated *lazily* by ``get_model_class``.

    Rather than requiring a particular layout, we probe for whichever
    capabilities the checkout in use actually provides and adapt.
    """

    def __init__(self, model_base, module=None, package=None):
        self.model_base = model_base
        self.module = module      # legacy monolithic convert_hf_to_gguf module
        self.package = package    # `conversion` package (llama.cpp#17114+)

    # -- capability probing -------------------------------------------------

    def _lookup(self, name):
        """Find a helper by name, preferring the `conversion` package."""
        for source in (self.package, self.module):
            if source is not None:
                obj = getattr(source, name, None)
                if obj is not None:
                    return obj
        return None

    def _import_submodule(self, name):
        """Import `conversion.<name>` (new layout) or fall back to the monolith."""
        if self.package is not None:
            return importlib.import_module(f"{self.package.__name__}.{name}")
        return self.module

    @property
    def layout(self):
        return "conversion-package" if self.package is not None else "monolithic"

    @property
    def has_lazy_registry(self):
        """True when upstream exposes get_model_class(), which imports on demand."""
        return self._lookup("get_model_class") is not None

    # -- adapted operations -------------------------------------------------

    def load_hparams(self, dir_model, is_mistral_format=False):
        """Call ModelBase.load_hparams across both signature generations."""
        load = self.model_base.load_hparams
        try:
            accepts_mistral = "is_mistral_format" in inspect.signature(load).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins/C functions
            accepts_mistral = False

        if accepts_mistral:
            return load(dir_model, is_mistral_format)

        if is_mistral_format:
            raise ValueError(
                "This llama.cpp checkout predates Mistral-format support "
                "(added by llama.cpp#14737, 2025-08-11). Update llama.cpp or "
                "drop --mistral-format."
            )
        return load(dir_model)

    def model_architecture(self, hparams, mmproj=False):
        """Resolve the architecture string using upstream's own resolver.

        Upstream's ``get_model_architecture`` understands nested layouts that a
        bare ``hparams["architectures"][0]`` misses — ``text_config``, InternVL's
        ``llm_config``, Qwen2.5-Omni's ``thinker_config``, DeepSeek-OCR's
        ``language_config`` and non-HF Mamba's ``ssm_cfg``.
        """
        resolver = self._lookup("get_model_architecture")
        if resolver is not None:
            model_type = self._model_type(mmproj)
            if model_type is not None:
                try:
                    return resolver(hparams, model_type)
                except TypeError:
                    pass
            return resolver(hparams)

        # Very old checkouts predate get_model_architecture.
        architectures = hparams.get("architectures")
        if not architectures:
            raise ValueError(
                "Could not determine the model architecture: no 'architectures' "
                "key in the model config."
            )
        return architectures[0]

    def _model_type(self, mmproj=False):
        model_type = self._lookup("ModelType")
        if model_type is None:
            return None
        return model_type.MMPROJ if mmproj else model_type.TEXT

    def model_class(self, architecture, mmproj=False):
        """Resolve architecture -> model class, importing lazily when required.

        Post-llama.cpp#17114 ``ModelBase._model_classes`` starts *empty*: classes
        register themselves only when their ``conversion.<family>`` module is
        imported. ``from_model_architecture`` is a plain registry read and so
        resolves nothing, which is why this tool previously reported almost every
        model as unsupported. ``get_model_class`` performs that import first.
        """
        get_model_class = self._lookup("get_model_class")
        if get_model_class is not None:
            return get_model_class(architecture, mmproj=mmproj)

        # Legacy monolith: importing the module registered every class eagerly.
        model_type = self._model_type(mmproj)
        if model_type is not None:
            try:
                return self.model_base.from_model_architecture(architecture, model_type)
            except TypeError:
                pass
        return self.model_base.from_model_architecture(architecture)

    def mistral_model_class(self, hparams, mmproj=False):
        """Select the Mistral-native class the way upstream's CLI does."""
        if mmproj:
            if hparams.get("vision_encoder") is None:
                raise ValueError("This model does not support multimodal conversion")
            return getattr(self._import_submodule("pixtral"), "PixtralModel")

        mistral = self._import_submodule("mistral")
        name = "MistralMoeModel" if hparams.get("moe") is not None else "MistralModel"
        model_class = getattr(mistral, name, None)
        if model_class is None:
            raise ValueError(
                f"This llama.cpp checkout does not provide {name}; "
                "update llama.cpp to convert Mistral-format models."
            )
        return model_class

    def require_mistral_common(self):
        """Mistral-format conversion needs the `mistral-common` package."""
        installed = self._lookup("_mistral_common_installed")
        if installed is False:
            message = self._lookup("_mistral_import_error_msg") or (
                "Mistral format requires the `mistral-common` package: "
                "pip install mistral-common"
            )
            raise ImportError(message)


def setup_llama_cpp_path(llama_cpp_dir=None):
    """Set up the llama.cpp path and import necessary modules"""
    global LLAMA_CPP_PATH, Model, UPSTREAM
    
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
        convert_module = None
        package = None

        # Post-llama.cpp#17114 the converter lives in a `conversion` package.
        # Prefer it: it owns the lazy model registry and the architecture
        # resolver, so we never have to guess at class names.
        try:
            package = importlib.import_module("conversion")
        except ImportError:
            package = None
        else:
            # `conversion` is a generic name; make sure we imported llama.cpp's.
            package_file = getattr(package, "__file__", None) or ""
            if not str(Path(package_file).resolve()).startswith(str(LLAMA_CPP_PATH.resolve())):
                logger.debug("Ignoring unrelated `conversion` package at %s", package_file)
                package = None

        if package is None:
            # Legacy monolithic convert_hf_to_gguf.py. Import it by file path so
            # its __main__ block does not run.
            convert_script_path = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"
            if not convert_script_path.exists():
                raise FileNotFoundError(f"Could not find {convert_script_path}")

            spec = importlib.util.spec_from_file_location(
                "convert_hf_to_gguf",
                str(convert_script_path)
            )
            convert_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(convert_module)

        # Resolve the base class. It has been `ModelBase` since ~2025-04-30;
        # `Model` is only present on checkouts older than that.
        global Model, UPSTREAM
        source = package if package is not None else convert_module
        Model = getattr(source, "ModelBase", None) or getattr(source, "Model", None)
        if Model is None:
            raise ImportError(
                "Incompatible llama.cpp: could not find a `ModelBase` (or legacy "
                "`Model`) class in convert_hf_to_gguf.py or the `conversion` "
                "package. Point --llama-cpp-dir at a llama.cpp checkout, or open "
                "an issue with the llama.cpp commit hash."
            )

        UPSTREAM = UpstreamConverter(Model, module=convert_module, package=package)
        logger.debug(
            "llama.cpp converter layout: %s (lazy registry: %s)",
            UPSTREAM.layout, UPSTREAM.has_lazy_registry,
        )

        return LLAMA_CPP_PATH
    except ImportError:
        # Already a clear, actionable message — propagate it verbatim.
        raise
    except Exception as e:
        raise ImportError(f"Error importing llama.cpp conversion modules: {e}")

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
        help="Path to write the output GGUF file (default: same directory as the model with model name and .gguf extension)"
    )
    
    parser.add_argument(
        "--outtype", type=str, choices=[
            # Full precision
            "f32", "f16", "bf16",
            # Single quantization type supported directly by the converter
            "q8_0",
            # Ternary quantization
            "tq1_0", "tq2_0",
            # Auto detection
            "auto"
        ], default="auto",
        help="Output data type for conversion (default: auto). For k-quants "
             "(q4_k, q5_k, ...) and other formats, first convert to f16/f32 here, "
             "then run quantize_gguf.py on the result."
    )
    
    parser.add_argument(
        "--bigendian", action="store_true",
        help="Use big endian format for output file (default: little endian / x86)"
    )
    
    parser.add_argument(
        "--optimize-for-size", action="store_true",
        help="Optimize the conversion for smaller file size (may reduce precision)"
    )
    
    parser.add_argument(
        "--optimize-output-tensor", action="store_true",
        help="Apply special optimization to the output tensor (may reduce size)"
    )
    
    parser.add_argument(
        "--optimize-token-embeddings", action="store_true",
        help="Apply special optimization to the token embeddings (may reduce size)"
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

    parser.add_argument(
        "--mistral-format", action="store_true", default=None,
        help="Treat the model as Mistral-native (params.json + consolidated*.safetensors) "
             "rather than HuggingFace format. Auto-detected when the directory has a "
             "params.json and no config.json. Requires the `mistral-common` package."
    )

    parser.add_argument(
        "--no-mistral-format", dest="mistral_format", action="store_false",
        help="Disable Mistral-format auto-detection and force HuggingFace format."
    )

    return parser.parse_args()


def resolve_mistral_format(args) -> bool:
    """Decide whether to read the model as Mistral-native.

    Explicit --mistral-format / --no-mistral-format always win. Otherwise a
    directory carrying params.json but no config.json is Mistral-native: that is
    the layout llama.cpp's own `--mistral-format` expects.
    """
    if args.mistral_format is not None:
        return args.mistral_format

    has_params = (args.model / "params.json").is_file()
    has_config = (args.model / "config.json").is_file()
    detected = has_params and not has_config
    if detected:
        logger.info(
            "Detected Mistral-native layout (params.json, no config.json); "
            "converting with --mistral-format. Pass --no-mistral-format to override."
        )
    return detected

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
    logger = logging.getLogger("safetensors-to-gguf")
    
    # Log optimization settings if enabled
    if args.optimize_for_size:
        logger.info("Optimizing for size: enabled")
    if args.optimize_output_tensor:
        logger.info("Output tensor optimization: enabled")
    if args.optimize_token_embeddings:
        logger.info("Token embeddings optimization: enabled")
    
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
    
    # Mistral-native models ship params.json instead of an HF config.json.
    is_mistral_format = resolve_mistral_format(args)
    if is_mistral_format:
        UPSTREAM.require_mistral_common()

    # Load model hyperparameters
    logger.info(f"Loading model: {args.model.name}")
    hparams = UPSTREAM.load_hparams(args.model, is_mistral_format)

    # Debug: Print the hyperparameters structure
    if args.verbose:
        logger.debug(f"Model hyperparameters: {json.dumps(hparams, indent=2, default=str)}")
        if "text_config" in hparams:
            logger.debug(f"text_config: {json.dumps(hparams['text_config'], indent=2, default=str)}")
            logger.debug(f"num_hidden_layers: {hparams['text_config'].get('num_hidden_layers')}")

    try:
        # Resolve the model class. Both steps delegate to llama.cpp so that every
        # architecture it supports is available here too — and so that a future
        # upstream refactor does not silently strand this tool again.
        if is_mistral_format:
            # params.json has no "architectures"; upstream selects the class from
            # the payload shape instead.
            model_class = UPSTREAM.mistral_model_class(hparams)
            logger.info(f"Mistral format detected; using model class: {model_class.__name__}")
        else:
            model_architecture = UPSTREAM.model_architecture(hparams)
            logger.info(f"Model architecture: {model_architecture}")
            try:
                model_class = UPSTREAM.model_class(model_architecture)
            except NotImplementedError:
                logger.error(f"Model {model_architecture} is not supported by llama.cpp")
                sys.exit(1)
            logger.info(f"Using model class: {model_class.__name__}")

        # Create model instance
        import torch
        with torch.inference_mode():
            logger.info("Creating model instance...")
            
            # Ensure we have a valid output filename
            outfile = args.outfile
            if outfile is None:
                # Generate a default output filename in the same directory as the model
                model_name = args.model_name or args.model.name
                # Use the model directory as the output directory
                outfile = args.model / f"{model_name}.gguf"
                logger.info(f"No output file specified, writing to model directory: {outfile}")
            
            # Set optimization flags in hparams for the model to use
            if args.optimize_for_size or args.optimize_output_tensor or args.optimize_token_embeddings:
                hparams["optimize_for_size"] = args.optimize_for_size
                hparams["optimize_output_tensor"] = args.optimize_output_tensor
                hparams["optimize_token_embeddings"] = args.optimize_token_embeddings
            
            model_instance = model_class(
                args.model, 
                output_type, 
                outfile,
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
