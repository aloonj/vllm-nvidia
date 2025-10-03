#!/usr/bin/env python3
"""
Multi-GPU Configuration Check for vLLM
Verifies and displays GPU setup for tensor parallelism
"""

import os
import torch
import warnings

# Suppress any deprecation warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

# Import NVIDIA ML library at module level (nvidia-ml-py3 provides pynvml interface)
nvml_available = False
try:
    from pynvml import *
    nvmlInit()
    nvml_available = True
    print("NVIDIA ML library initialized successfully")
except ImportError:
    print("Warning: NVIDIA ML library not available. GPU monitoring limited.")
except Exception as e:
    print(f"Warning: NVIDIA ML initialization failed: {e}")

def setup_multi_gpu():
    """Configure multi-GPU setup for vLLM"""
    # Set CUDA device order
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Enable tensor parallelism across GPUs
    os.environ["VLLM_USE_MODELSCOPE"] = "False"

    # Verify GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory // 1024**3}GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")

            # Try to get detailed memory info if nvml is available
            if nvml_available:
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    print(f"  Available Memory: {mem_info.free // 1024**3}GB")
                    print(f"  Used Memory: {mem_info.used // 1024**3}GB")
                except Exception as e:
                    print(f"  Memory details unavailable: {e}")
    else:
        print("CUDA not available!")
        return 0

    return gpu_count

if __name__ == "__main__":
    gpu_count = setup_multi_gpu()
    print(f"\nRecommended tensor_parallel_size: {gpu_count}")