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
    "model": "RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.95,
    "max_model_len": 32768,  # try this first
    "dtype": "bfloat16",
    "kv_cache_dtype": "fp8",  # KEY ADDITION
    "trust_remote_code": True,
    "disable_custom_all_reduce": True,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 16,  # reduce if OOM
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
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
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"],
                       help="Model to load")
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_CONFIG["tensor_parallel_size"],
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_CONFIG["gpu_memory_utilization"],
                       help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_CONFIG["max_model_len"],
                       help="Maximum model context length")
    parser.add_argument("--dtype", default=DEFAULT_CONFIG["dtype"],
                       choices=["float16", "bfloat16", "float32"],
                       help="Model data type")

    # Server configuration
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"],
                       help="Server port")
    parser.add_argument("--host", default=DEFAULT_CONFIG["host"],
                       help="Server host")

    # Performance options
    parser.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_CONFIG["max_num_batched_tokens"],
                       help="Maximum number of batched tokens")

    # Boolean flags
    parser.add_argument("--no-trust-remote-code", action="store_true",
                       help="Disable trust_remote_code")
    parser.add_argument("--no-disable-custom-all-reduce", action="store_true",
                       help="Enable custom all reduce")
    parser.add_argument("--no-prefix-caching", action="store_true",
                       help="Disable prefix caching")

    # Preset configurations
    parser.add_argument("--preset", choices=["gemma27b", "gemma12b", "llama13b"],
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
    elif args.preset == "gemma12b":
        config.update({
            "model": "google/gemma-2-12b-it",
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
            "dtype": "bfloat16"
        })
    elif args.preset == "llama13b":
        config.update({
            "model": "meta-llama/Llama-2-13b-chat-hf",
            "tensor_parallel_size": 2,
            "max_model_len": 4096,
            "dtype": "float16"
        })

    # Apply command line arguments
    config["model"] = args.model
    config["tensor_parallel_size"] = args.tensor_parallel_size
    config["gpu_memory_utilization"] = args.gpu_memory_utilization
    config["max_model_len"] = args.max_model_len
    config["dtype"] = args.dtype
    config["port"] = args.port
    config["host"] = args.host
    config["max_num_batched_tokens"] = args.max_num_batched_tokens

    # Handle boolean flags
    config["trust_remote_code"] = not args.no_trust_remote_code
    config["disable_custom_all_reduce"] = not args.no_disable_custom_all_reduce
    config["enable_prefix_caching"] = not args.no_prefix_caching

    # Launch server
    launch_vllm_server(config)

if __name__ == "__main__":
    main()
