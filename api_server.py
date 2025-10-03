#!/usr/bin/env python3
"""
vLLM OpenAI-Compatible Server Launcher
Configurable Python wrapper for starting vLLM's OpenAI server
"""

import subprocess
import sys
import argparse
import os

# Default configuration
DEFAULT_CONFIG = {
    "port": 8000,
    "host": "0.0.0.0"
}

def launch_vllm_server(config=None):
    """Launch vLLM OpenAI server with specified configuration"""

    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Build command line arguments
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model"],
        "--tensor-parallel-size", str(config["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-model-len", str(config["max_model_len"]),
        "--dtype", config["dtype"],
        "--port", str(config["port"]),
        "--host", config["host"]
    ]

    # Add boolean flags
    if config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    if config.get("disable_custom_all_reduce"):
        cmd.append("--disable-custom-all-reduce")

    if config.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")

    # Add optional parameters
    if config.get("max_num_batched_tokens"):
        cmd.extend(["--max-num-batched-tokens", str(config["max_num_batched_tokens"])])

    if config.get("max_num_seqs"):
        cmd.extend(["--max-num-seqs", str(config["max_num_seqs"])])

    if config.get("kv_cache_dtype") and config["kv_cache_dtype"] is not None:
        cmd.extend(["--kv-cache-dtype", config["kv_cache_dtype"]])

    if config.get("enable_chunked_prefill"):
        cmd.append("--enable-chunked-prefill")

    print("Starting vLLM OpenAI Server with configuration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        # Launch the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI-Compatible Server")

    # Model configuration
    parser.add_argument("--model", default=None,
                       help="Model to load")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                       help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=None,
                       help="Maximum model context length")
    parser.add_argument("--dtype", default=None,
                       choices=["float16", "bfloat16", "float32"],
                       help="Model data type")

    # Server configuration
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"],
                       help="Server port")
    parser.add_argument("--host", default=DEFAULT_CONFIG["host"],
                       help="Server host")

    # Performance options
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                       help="Maximum number of batched tokens")

    # Boolean flags
    parser.add_argument("--no-trust-remote-code", action="store_true",
                       help="Disable trust_remote_code")
    parser.add_argument("--no-disable-custom-all-reduce", action="store_true",
                       help="Enable custom all reduce")
    parser.add_argument("--no-prefix-caching", action="store_true",
                       help="Disable prefix caching")

    # Preset configurations
    parser.add_argument("--preset", choices=["gemma27b", "gemma12b", "llama13b", "qwen30b"],
                       help="Use preset configuration")

    args = parser.parse_args()

    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Apply preset configurations
    if args.preset == "gemma27b":
        config.update({
            "model": "RedHatAI/gemma-3-27b-it-quantized.w4a16",
            "tensor_parallel_size": 2,
            "max_model_len": 16384,
            "dtype": "bfloat16"
        })
    elif args.preset == "qwen30b":
        config.update({
            "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",  # GPTQ 4-bit (better than FP8 for 3090s)
            "quantization": "gptq",  # CRITICAL: Specify GPTQ quantization
            "tensor_parallel_size": 2,  # Use both GPUs
            "gpu_memory_utilization": 0.95,  # Use 95% of 48GB
            "max_model_len": 40960,
            "dtype": "auto",  # Auto-detect precision
            "kv_cache_dtype": None,  # Disable FP8 KV cache for qwen30b
            "max_num_batched_tokens": 65536,  # Larger for chunked prefill
            "max_num_seqs": 4,  # 4 concurrent sequences
            "enable_prefix_caching": True,  # Cache common prefixes
            "enable_chunked_prefill": True,  # Process long prompts in chunks
            "chunked_prefill_size": 8192  # Chunk size for prefill
        })

    # Only apply command line arguments if they were explicitly provided (not defaults)
    # Check if argument was actually passed by user
    import sys
    provided_args = sys.argv[1:]

    if '--model' in provided_args:
        config["model"] = args.model
    if '--tensor-parallel-size' in provided_args:
        config["tensor_parallel_size"] = args.tensor_parallel_size
    if '--gpu-memory-utilization' in provided_args:
        config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if '--max-model-len' in provided_args:
        config["max_model_len"] = args.max_model_len
    if '--dtype' in provided_args:
        config["dtype"] = args.dtype
    if '--port' in provided_args:
        config["port"] = args.port
    if '--host' in provided_args:
        config["host"] = args.host
    if '--max-num-batched-tokens' in provided_args:
        config["max_num_batched_tokens"] = args.max_num_batched_tokens

    # Handle boolean flags
    if '--no-trust-remote-code' in provided_args:
        config["trust_remote_code"] = False
    if '--no-disable-custom-all-reduce' in provided_args:
        config["disable_custom_all_reduce"] = False
    if '--no-prefix-caching' in provided_args:
        config["enable_prefix_caching"] = False

    # Launch server
    launch_vllm_server(config)

if __name__ == "__main__":
    main()
