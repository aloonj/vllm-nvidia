#!/usr/bin/env python3
"""
vLLM OpenAI-Compatible Server Launcher
Supports external YAML profiles for configuration
"""

import subprocess
import sys
import argparse
import os
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "port": 8000,
    "host": "0.0.0.0"
}

def detect_gpu_count():
    """Detect the number of available GPUs"""
    try:
        # Try nvidia-smi first (most reliable)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_count = len(result.stdout.strip().split('\n'))
        if gpu_count > 0:
            print(f"Detected {gpu_count} GPU(s) via nvidia-smi")
            return gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        # Fallback to torch if available
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                print(f"Detected {gpu_count} GPU(s) via PyTorch")
                return gpu_count
    except ImportError:
        pass

    print("Warning: Could not detect GPUs, defaulting to 1")
    return 1

def load_profile(profile_path):
    """Load a profile from a YAML file"""
    try:
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
            print(f"Loaded profile: {profile.get('name', 'unnamed')} - {profile.get('description', '')}")
            return profile
    except Exception as e:
        print(f"Error loading profile {profile_path}: {e}")
        sys.exit(1)

def find_profile(profile_name):
    """Find a profile file by name in common locations"""
    search_paths = [
        Path.cwd() / "profiles",
        Path.home() / ".vllm" / "profiles",
        Path(__file__).parent / "profiles"
    ]

    for path in search_paths:
        if not path.exists():
            continue

        profile_file = path / f"{profile_name}.yaml"
        if profile_file.exists():
            return profile_file

        profile_file = path / f"{profile_name}.yml"
        if profile_file.exists():
            return profile_file

    return None

def list_profiles():
    """List all available profiles"""
    profiles = {}
    search_paths = [
        Path.cwd() / "profiles",
        Path.home() / ".vllm" / "profiles",
        Path(__file__).parent / "profiles"
    ]

    for path in search_paths:
        if not path.exists():
            continue

        for file in path.glob("*.yaml"):
            if file.stem not in profiles:
                try:
                    with open(file, 'r') as f:
                        data = yaml.safe_load(f)
                        profiles[file.stem] = {
                            'path': file,
                            'description': data.get('description', 'No description')
                        }
                except:
                    pass

        for file in path.glob("*.yml"):
            if file.stem not in profiles:
                try:
                    with open(file, 'r') as f:
                        data = yaml.safe_load(f)
                        profiles[file.stem] = {
                            'path': file,
                            'description': data.get('description', 'No description')
                        }
                except:
                    pass

    return profiles

def launch_vllm_server(config=None):
    """Launch vLLM OpenAI server with specified configuration"""

    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Handle 'auto' tensor_parallel_size
    tensor_parallel_size = config.get("tensor_parallel_size", 1)
    if str(tensor_parallel_size).lower() == "auto":
        tensor_parallel_size = detect_gpu_count()
        config["tensor_parallel_size"] = tensor_parallel_size
    else:
        # Validate tensor_parallel_size
        try:
            tensor_parallel_size = int(tensor_parallel_size)
            available_gpus = detect_gpu_count()
            if tensor_parallel_size > available_gpus:
                print(f"Warning: tensor_parallel_size ({tensor_parallel_size}) exceeds available GPUs ({available_gpus})")
                print(f"This may cause errors. Consider using 'auto' or reducing to {available_gpus}")
        except (ValueError, TypeError):
            print(f"Error: Invalid tensor_parallel_size '{tensor_parallel_size}'")
            sys.exit(1)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model"],
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
        "--max-model-len", str(config["max_model_len"]),
        "--dtype", config["dtype"],
        "--port", str(config.get("port", DEFAULT_CONFIG["port"])),
        "--host", config.get("host", DEFAULT_CONFIG["host"])
    ]

    if config.get("quantization"):
        cmd.extend(["--quantization", config["quantization"]])

    if config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    if config.get("disable_custom_all_reduce"):
        cmd.append("--disable-custom-all-reduce")

    if config.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")

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
        if key not in ['name', 'description']:
            print(f"{key}: {value}")
    print("-" * 50)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI-Compatible Server")

    parser.add_argument("--model", default=None, help="Model to load")
    parser.add_argument("--tensor-parallel-size", default=None,
                       help="Number of GPUs for tensor parallelism (use 'auto' to detect)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                       help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=None,
                       help="Maximum model context length")
    parser.add_argument("--dtype", default=None,
                       choices=["float16", "bfloat16", "float32", "auto"],
                       help="Model data type")

    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"],
                       help="Server port")
    parser.add_argument("--host", default=DEFAULT_CONFIG["host"],
                       help="Server host")

    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                       help="Maximum number of batched tokens")

    parser.add_argument("--no-trust-remote-code", action="store_true",
                       help="Disable trust_remote_code")
    parser.add_argument("--no-disable-custom-all-reduce", action="store_true",
                       help="Enable custom all reduce")
    parser.add_argument("--no-prefix-caching", action="store_true",
                       help="Disable prefix caching")

    parser.add_argument("--profile", type=str, default=None,
                       help="Profile name or path to YAML profile file")
    parser.add_argument("--list-profiles", action="store_true",
                       help="List all available profiles")

    args = parser.parse_args()

    if args.list_profiles:
        profiles = list_profiles()
        if profiles:
            print("Available profiles:")
            print("-" * 50)
            for name, info in profiles.items():
                print(f"{name:15} - {info['description']}")
                print(f"{'':15}   Path: {info['path']}")
        else:
            print("No profiles found. Create profiles in:")
            print("  - ./profiles/")
            print("  - ~/.vllm/profiles/")
        sys.exit(0)

    config = DEFAULT_CONFIG.copy()

    if args.profile:
        if args.profile.endswith(('.yaml', '.yml')) and os.path.exists(args.profile):
            profile_config = load_profile(args.profile)
        else:
            profile_path = find_profile(args.profile)
            if profile_path:
                profile_config = load_profile(profile_path)
            else:
                print(f"Profile '{args.profile}' not found")
                print("Use --list-profiles to see available profiles")
                sys.exit(1)

        config.update(profile_config)

    provided_args = sys.argv[1:]

    if '--model' in provided_args and args.model:
        config["model"] = args.model
    if '--tensor-parallel-size' in provided_args:
        # Handle 'auto' from command line
        if str(args.tensor_parallel_size).lower() == "auto":
            config["tensor_parallel_size"] = "auto"
        else:
            try:
                config["tensor_parallel_size"] = int(args.tensor_parallel_size)
            except ValueError:
                print(f"Error: Invalid tensor-parallel-size '{args.tensor_parallel_size}'. Use a number or 'auto'")
                sys.exit(1)
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

    if '--no-trust-remote-code' in provided_args:
        config["trust_remote_code"] = False
    if '--no-disable-custom-all-reduce' in provided_args:
        config["disable_custom_all_reduce"] = False
    if '--no-prefix-caching' in provided_args:
        config["enable_prefix_caching"] = False

    required_fields = ["model", "tensor_parallel_size", "gpu_memory_utilization",
                      "max_model_len", "dtype"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        print(f"Error: Missing required configuration: {', '.join(missing_fields)}")
        print("Provide these via profile or command line arguments")
        sys.exit(1)

    launch_vllm_server(config)

if __name__ == "__main__":
    main()