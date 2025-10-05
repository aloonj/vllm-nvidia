#!/usr/bin/env python3
"""
vRAM Estimation Tool for vLLM Model Profiles

Estimates vRAM requirements for models based on:
- Model size (parameters)
- Quantization type
- Context length (KV cache)
- Tensor parallelism
- Batch size and other factors
"""

import os
import yaml
import re
import argparse
import subprocess
import json
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple

# Cache for HF API responses to avoid repeated requests
_hf_size_cache = {}

# Model size estimates (in billions of parameters)
MODEL_SIZES = {
    # Qwen models
    'qwen3-30b': 30.0,
    'qwen3-coder-30b': 30.0,
    'qwen3-vl-30b': 30.0,

    # Gemma models
    'gemma-3-27b': 27.0,
    'gemma-2-27b': 27.0,

    # Mistral models
    'magistral-small': 23.3,  # Actual from config.json
    'devstral-small': 23.3,   # Actual from config.json (44GB file / 2 bytes)

    # LLaMA models
    'llama-2-13b': 13.0,
    'llama-2-7b': 7.0,
    'llama-3-8b': 8.0,
    'llama-3-70b': 70.0,
}

# Quantization memory multipliers (relative to FP16)
QUANTIZATION_MULTIPLIERS = {
    'fp16': 1.0,
    'bfloat16': 1.0,
    'fp8': 0.5,
    'int8': 0.5,
    'int4': 0.25,
    'gptq': 0.25,  # GPTQ typically uses 4-bit
    'awq': 0.25,   # AWQ typically uses 4-bit
    'auto': 1.0,   # Assume no quantization
    None: 1.0,
}

def detect_gpu_info():
    """Detect GPU count and memory per GPU"""
    try:
        # Get GPU count
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_count = len(gpu_result.stdout.strip().split('\n'))

        # Get GPU memory (in MB)
        memory_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        memory_mb = int(memory_result.stdout.strip().split('\n')[0])
        memory_gb = memory_mb / 1024

        if gpu_count > 0:
            return gpu_count, memory_gb
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    try:
        # Fallback to torch if available
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Get memory from first GPU (assume all are same)
                memory_bytes = torch.cuda.get_device_properties(0).total_memory
                memory_gb = memory_bytes / (1024**3)
                return gpu_count, memory_gb
    except ImportError:
        pass

    # Default fallback
    print("Warning: Could not detect GPUs, assuming 1 GPU with 24GB")
    return 1, 24.0

def get_hf_model_size(model_id: str) -> Optional[float]:
    """Get model size from HuggingFace API without downloading."""
    # Check cache first
    if model_id in _hf_size_cache:
        return _hf_size_cache[model_id]

    files_url = f"https://huggingface.co/api/models/{model_id}/tree/main"

    try:
        response = requests.get(files_url, timeout=5)
        if response.status_code == 200:
            files = response.json()
            consolidated_size = 0
            sharded_size = 0
            other_model_size = 0

            for file_info in files:
                if isinstance(file_info, dict) and 'size' in file_info:
                    path = file_info.get('path', '')

                    if any(path.endswith(ext) for ext in ['.safetensors', '.bin', '.pt', '.pth']):
                        size = file_info['size']
                        if 'consolidated' in path.lower():
                            # Prefer consolidated file
                            consolidated_size = max(consolidated_size, size)
                        elif 'model-' in path and '-of-' in path:
                            # Sum sharded files
                            sharded_size += size
                        elif path.startswith('model.') or path.startswith('pytorch_model.'):
                            # Single model file (e.g., model.safetensors, pytorch_model.bin)
                            other_model_size = max(other_model_size, size)

            # Use consolidated if available, then sharded, then single model file
            if consolidated_size > 0:
                total_size = consolidated_size
            elif sharded_size > 0:
                total_size = sharded_size
            else:
                total_size = other_model_size

            if total_size > 0:
                size_gb = total_size / (1024**3)
                _hf_size_cache[model_id] = size_gb  # Cache the result
                return size_gb
    except:
        pass

    _hf_size_cache[model_id] = None  # Cache the failure too
    return None


def get_model_params_from_config(model_name: str) -> Optional[float]:
    """Try to get actual model parameters from HuggingFace cache config.json."""
    # Convert model name to HF cache format
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'

    # Try different cache name formats
    model_id = model_name.replace('/', '--')
    possible_dirs = [
        cache_dir / f'models--{model_id}',
        cache_dir / f'models--{model_id.lower()}',
    ]

    for model_dir in possible_dirs:
        if model_dir.exists():
            # Look for config.json in snapshots
            config_files = list(model_dir.glob('snapshots/*/config.json'))
            if config_files:
                try:
                    with open(config_files[0], 'r') as f:
                        config = json.load(f)

                    # Calculate parameters based on architecture
                    model_type = config.get('model_type', '')
                    hidden_size = config.get('hidden_size', 0)
                    num_layers = config.get('num_hidden_layers', 0)
                    vocab_size = config.get('vocab_size', 0)
                    intermediate_size = config.get('intermediate_size', hidden_size * 4)
                    num_kv_heads = config.get('num_key_value_heads', config.get('num_attention_heads', 0))
                    head_dim = config.get('head_dim', hidden_size // config.get('num_attention_heads', 1) if config.get('num_attention_heads') else 128)

                    # Handle MoE models differently
                    if 'moe' in model_type.lower():
                        # For MoE, use pre-calculated sizes or estimate from file size
                        file_size = get_actual_model_size(model_name)
                        if file_size:
                            # Estimate params from file size (assuming BF16 or FP8)
                            if 'fp8' in model_name.lower():
                                params_billions = file_size  # FP8 = 1 byte per param
                            else:
                                params_billions = file_size / 2  # BF16 = 2 bytes per param
                            return params_billions
                        # Fallback for known MoE models
                        if 'qwen3-30b' in model_name.lower():
                            return 30.0

                    if hidden_size and num_layers and vocab_size:
                        # Calculate total parameters
                        embeddings = vocab_size * hidden_size

                        # Attention layers (with GQA support)
                        attention_q = num_layers * hidden_size * hidden_size
                        attention_kv = num_layers * hidden_size * (num_kv_heads * head_dim * 2)
                        attention_o = num_layers * hidden_size * hidden_size
                        attention_total = attention_q + attention_kv + attention_o

                        # MLP layers
                        mlp_gate_up = num_layers * hidden_size * intermediate_size * 2  # gate and up
                        mlp_down = num_layers * intermediate_size * hidden_size
                        mlp_total = mlp_gate_up + mlp_down

                        # LayerNorms
                        layernorm = num_layers * 2 * hidden_size

                        total_params = embeddings + attention_total + mlp_total + layernorm
                        params_billions = total_params / 1e9

                        # print(f"  Found config for {model_name}: {params_billions:.1f}B params")
                        return params_billions
                except Exception as e:
                    pass  # Silently skip parse errors

    return None

def estimate_model_params(model_name: str) -> float:
    """Estimate model parameters in billions based on model name or config."""
    # First try to get actual params from config
    actual_params = get_model_params_from_config(model_name)
    if actual_params:
        return actual_params

    model_lower = model_name.lower()

    # Try to extract size from model name
    size_patterns = [
        r'(\d+)b',  # e.g., "30b", "27b"
        r'(\d+\.?\d*)b',  # e.g., "7.5b"
    ]

    for pattern in size_patterns:
        match = re.search(pattern, model_lower)
        if match:
            return float(match.group(1))

    # Fallback to known model sizes
    for key, size in MODEL_SIZES.items():
        if key in model_lower:
            return size

    # Default estimate if unknown
    print(f"Warning: Unknown model size for {model_name}, assuming 30B parameters")
    return 30.0

def detect_quantization(model_name: str, config_quantization: str, dtype: str) -> str:
    """Detect quantization type from model name and config."""
    model_lower = model_name.lower()

    # Check config first
    if config_quantization:
        return config_quantization

    # Detect from model name
    if 'fp8' in model_lower:
        return 'fp8'
    elif 'awq' in model_lower:
        return 'awq'
    elif 'gptq' in model_lower:
        return 'gptq'
    elif 'int4' in model_lower:
        return 'int4'
    elif 'int8' in model_lower:
        return 'int8'

    return dtype or 'auto'

def estimate_model_memory(
    params_billions: float,
    quantization: Optional[str] = None,
    dtype: str = 'auto',
    model_name: str = ''
) -> float:
    """Estimate base model memory in GB."""

    # Base memory calculation (FP16 = 2 bytes per parameter)
    base_memory_gb = params_billions * 2.0  # GB

    # Special case for Mistral models which are often stored in larger format
    if 'mistral' in model_name.lower() or 'devstral' in model_name.lower() or 'magistral' in model_name.lower():
        if dtype == 'auto' and not quantization:
            # Mistral models default to BF16/FP32 storage (44GB for 22B model)
            base_memory_gb = params_billions * 2.0  # Already correct for BF16

    # Apply quantization multiplier
    quant_type = quantization or dtype
    multiplier = QUANTIZATION_MULTIPLIERS.get(quant_type, 1.0)

    return base_memory_gb * multiplier

def estimate_kv_cache_memory(
    params_billions: float,
    max_model_len: int,
    max_num_seqs: int = 1,
    tensor_parallel_size: int = 1,
    quantization: str = 'auto'
) -> float:
    """Estimate KV cache memory in GB."""

    # Simplified KV cache estimation based on empirical observations
    # vLLM uses page-based allocation which is more efficient

    # Base MB per 1K tokens (empirically observed)
    if params_billions <= 7:
        mb_per_1k_tokens = 50  # ~50MB per 1K tokens for 7B models
    elif params_billions <= 13:
        mb_per_1k_tokens = 80  # ~80MB per 1K tokens for 13B models
    elif params_billions <= 22:
        mb_per_1k_tokens = 100  # ~100MB per 1K tokens for 22B models
    elif params_billions <= 30:
        mb_per_1k_tokens = 120  # ~120MB per 1K tokens for 30B models
    else:
        mb_per_1k_tokens = 150  # ~150MB per 1K tokens for larger models

    # Apply quantization factor for KV cache
    if quantization == 'fp8':
        mb_per_1k_tokens *= 0.5  # FP8 KV cache uses half memory
    elif quantization in ['gptq', 'awq', 'int4']:
        # Quantized models often still use FP16/BF16 KV cache
        mb_per_1k_tokens *= 1.0

    # Calculate total KV cache
    total_kv_mb = (max_model_len / 1000) * mb_per_1k_tokens * max_num_seqs

    # Convert to GB and distribute across tensor parallel devices
    kv_per_device_gb = (total_kv_mb / 1000) / tensor_parallel_size

    return kv_per_device_gb

def estimate_activation_memory(params_billions: float, max_num_seqs: int = 1) -> float:
    """Estimate activation memory in GB."""
    # Activation memory is relatively small compared to model weights
    # Includes intermediate tensors, gradients (if any), and workspace
    activation_per_seq_gb = min(params_billions * 0.02, 1.0)  # Cap at 1GB per sequence
    return activation_per_seq_gb * max_num_seqs

def analyze_profile(profile_path: str, detected_gpu_count: int = 2) -> Dict:
    """Analyze a single profile and estimate its vRAM requirements."""

    with open(profile_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration
    model_name = config.get('model', '')
    quantization = config.get('quantization')
    dtype = config.get('dtype', 'auto')
    max_model_len = config.get('max_model_len', 4096)
    max_num_seqs = config.get('max_num_seqs', 1)
    tensor_parallel_size = config.get('tensor_parallel_size', 1)
    gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)

    if tensor_parallel_size == 'auto':
        tensor_parallel_size = detected_gpu_count

    # Estimate model parameters
    params_billions = estimate_model_params(model_name)

    # Detect actual quantization type
    actual_quantization = detect_quantization(model_name, quantization, dtype)

    # Check for multimodal capabilities
    is_multimodal = bool(config.get('limit_mm_per_prompt')) or 'vl' in model_name.lower() or 'vision' in model_name.lower()

    # Get model size from HuggingFace API (most accurate)
    hf_size = get_hf_model_size(model_name)
    if hf_size:
        model_memory = hf_size
    else:
        model_memory = None  # Will be handled in display
    kv_cache_memory = estimate_kv_cache_memory(
        params_billions, max_model_len, max_num_seqs, tensor_parallel_size, actual_quantization
    )
    activation_memory = estimate_activation_memory(params_billions, max_num_seqs)

    # Add multimodal overhead if detected
    multimodal_overhead = 0
    if is_multimodal:
        # Mistral models have more overhead due to their architecture
        if 'mistral' in model_name.lower() or 'devstral' in model_name.lower() or 'magistral' in model_name.lower():
            multimodal_overhead = params_billions * 0.3  # 30% overhead for Mistral multimodal
        else:
            multimodal_overhead = params_billions * 0.1  # 10% for other multimodal models
        print(f"  Detected multimodal model - adding {multimodal_overhead:.1f}GB overhead")

    if model_memory is not None:
        # Total memory per GPU
        total_memory_per_gpu = (model_memory / tensor_parallel_size) + kv_cache_memory + activation_memory + multimodal_overhead

        # Apply real-world calibration based on actual vLLM measurements
        # Runtime memory includes overhead for CUDA kernels, memory fragmentation, etc.
        runtime_calibration_factor = 1.15  # 15% overhead for runtime (reduced for more accurate download prediction)

        # Loading requires temporary buffers but vLLM is efficient
        loading_spike_factor = 1.1  # 10% spike during loading

        # Calculate calibrated runtime memory
        calibrated_runtime = total_memory_per_gpu * runtime_calibration_factor

        # Calculate peak memory needed during loading
        estimated_required = calibrated_runtime * loading_spike_factor

        # Calculate what GPU memory utilization would allow
        max_allocatable = estimated_required / gpu_memory_utilization
    else:
        # No model size data available
        total_memory_per_gpu = None
        calibrated_runtime = None
        estimated_required = None
        max_allocatable = None

    return {
        'profile_name': config.get('name', Path(profile_path).stem),
        'model': model_name,
        'params_billions': params_billions,
        'quantization': actual_quantization,
        'max_model_len': max_model_len,
        'max_num_seqs': max_num_seqs,
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': gpu_memory_utilization,
        'model_memory_gb': model_memory,
        'kv_cache_memory_gb': kv_cache_memory,
        'activation_memory_gb': activation_memory,
        'total_per_gpu_gb': total_memory_per_gpu,
        'calibrated_runtime_gb': calibrated_runtime,
        'estimated_required_gb': estimated_required,
        'max_allocatable_gb': max_allocatable
    }

def format_memory(gb: float) -> str:
    """Format memory in GB with appropriate precision."""
    if gb < 1:
        return f"{gb*1024:.0f}MB"
    else:
        return f"{gb:.1f}GB"

def print_analysis(results: list, available_vram_gb: float = 24.0, show_per_gpu: bool = False):
    """Print formatted analysis results."""

    print(f"vRAM Estimation for Model Profiles")
    if show_per_gpu:
        print(f"Available vRAM per GPU: {format_memory(available_vram_gb)}")
    else:
        print(f"Available vRAM total: {format_memory(available_vram_gb)}")
    print("=" * 120)

    # Header
    print(f"{'Profile':<35} {'Model':<15} {'Params':<8} {'Quant':<8} {'Ctx':<8} {'TP':<4} "
          f"{'Model':<8} {'KV':<8} {'Act':<8} {'Runtime':<8} {'Peak':<8} {'Status':<10}")
    print("-" * 120)

    for result in sorted(results, key=lambda x: x['estimated_required_gb'] if x['estimated_required_gb'] is not None else float('inf')):
        name = result['profile_name'][:34]
        model = result['model'].split('/')[-1][:14]  # Just the model name part
        params = f"{result['params_billions']:.0f}B"
        quant = result['quantization'][:7] if result['quantization'] else 'none'
        ctx = f"{result['max_model_len']//1024}K" if result['max_model_len'] >= 1024 else str(result['max_model_len'])
        tp = str(result['tensor_parallel_size'])

        # Handle None values gracefully
        if result['model_memory_gb'] is not None:
            model_mem = format_memory(result['model_memory_gb'])[:7]
            kv_mem = format_memory(result['kv_cache_memory_gb'])[:7]
            act_mem = format_memory(result['activation_memory_gb'])[:7]
            runtime_mem = format_memory(result['calibrated_runtime_gb'])[:7]
            peak_mem = format_memory(result['estimated_required_gb'])[:7]

            # Status check
            if result['estimated_required_gb'] <= available_vram_gb:
                status = "✓ OK"
            elif result['max_allocatable_gb'] <= available_vram_gb:
                status = "⚠ Tight"
            else:
                status = "✗ OOM"
        else:
            model_mem = "No data"
            kv_mem = "-"
            act_mem = "-"
            runtime_mem = "-"
            peak_mem = "-"
            status = "❓ No HF data"

        print(f"{name:<35} {model:<15} {params:<8} {quant:<8} {ctx:<8} {tp:<4} "
              f"{model_mem:<8} {kv_mem:<8} {act_mem:<8} {runtime_mem:<8} {peak_mem:<8} {status:<10}")

def suggest_optimizations(result: Dict, available_vram_gb: float) -> list:
    """Suggest optimizations to fit within available vRAM."""
    suggestions = []

    if result['estimated_required_gb'] <= available_vram_gb:
        return suggestions

    # Calculate what max_model_len would fit
    model_memory_per_gpu = result['model_memory_gb'] / result['tensor_parallel_size']
    activation_memory = result['activation_memory_gb']
    available_for_kv = (available_vram_gb / 1.2) - model_memory_per_gpu - activation_memory

    if available_for_kv > 0:
        # Reverse engineer max_model_len from available KV cache memory
        params_billions = result['params_billions']
        max_num_seqs = result['max_num_seqs']

        # Estimate layers and attention dimensions
        if params_billions <= 7:
            num_layers, num_heads, head_dim = 32, 32, 128
        elif params_billions <= 13:
            num_layers, num_heads, head_dim = 40, 40, 128
        elif params_billions <= 30:
            num_layers, num_heads, head_dim = 60, 48, 128
        else:
            num_layers, num_heads, head_dim = 80, 64, 128

        # KV bytes per token = 2 * num_layers * num_heads * head_dim * 2 (FP16)
        kv_bytes_per_token = 2 * num_layers * num_heads * head_dim * 2

        # Available tokens = available_memory / (kv_bytes_per_token * max_num_seqs)
        available_tokens = (available_for_kv * 1024**3) / (kv_bytes_per_token * max_num_seqs)
        suggested_max_len = int(available_tokens)

        if suggested_max_len > 1024:
            suggestions.append(f"Reduce max_model_len to {suggested_max_len:,} tokens")

    # Suggest reducing max_num_seqs
    if result['max_num_seqs'] > 1:
        suggestions.append(f"Reduce max_num_seqs from {result['max_num_seqs']} to 1")

    # Suggest increasing tensor parallelism
    if result['tensor_parallel_size'] < 4:
        suggestions.append(f"Increase tensor_parallel_size to {result['tensor_parallel_size'] * 2}")

    # Suggest better quantization
    current_quant = result['quantization']
    if current_quant in ['auto', 'bfloat16', 'fp16']:
        suggestions.append("Use quantization (GPTQ, AWQ, or FP8) to reduce model memory")

    return suggestions

def main():
    parser = argparse.ArgumentParser(description="Estimate vRAM requirements for vLLM model profiles")
    parser.add_argument("--profiles-dir", default="profiles", help="Directory containing profile YAML files")
    parser.add_argument("--available-vram", type=float, help="Available vRAM per GPU in GB (auto-detected if not specified)")
    parser.add_argument("--profile", help="Analyze specific profile file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed breakdown")
    parser.add_argument("--suggest", "-s", action="store_true", help="Show optimization suggestions")
    parser.add_argument("--per-gpu", "-p", action="store_true", help="Show memory requirements per GPU instead of total")

    args = parser.parse_args()

    # Auto-detect GPU configuration
    detected_gpu_count, detected_vram_per_gpu = detect_gpu_info()
    total_vram = detected_gpu_count * detected_vram_per_gpu

    # Determine which memory limit to use
    if args.per_gpu:
        available_vram = args.available_vram if args.available_vram else detected_vram_per_gpu
        print(f"Detected: {detected_gpu_count} GPU(s) with {detected_vram_per_gpu:.1f}GB each (Total: {total_vram:.1f}GB)")
        print(f"Analyzing based on per-GPU memory ({detected_vram_per_gpu:.1f}GB)")
    else:
        # Use total memory for tensor parallel models by default
        available_vram = args.available_vram if args.available_vram else total_vram
        print(f"Detected: {detected_gpu_count} GPU(s) with {detected_vram_per_gpu:.1f}GB each (Total: {total_vram:.1f}GB)")
        print(f"Analyzing based on total available memory ({total_vram:.1f}GB) for tensor parallel models")

    profiles_dir = Path(args.profiles_dir)

    if args.profile:
        # Analyze single profile
        profile_path = args.profile
        if not os.path.exists(profile_path):
            profile_path = profiles_dir / f"{args.profile}.yaml"

        if not os.path.exists(profile_path):
            print(f"Profile not found: {args.profile}")
            return

        results = [analyze_profile(profile_path, detected_gpu_count)]
    else:
        # Analyze all profiles
        profile_files = list(profiles_dir.glob("*.yaml"))
        if not profile_files:
            print(f"No profile files found in {profiles_dir}")
            return

        results = []
        for profile_file in profile_files:
            if profile_file.name == "README.md":
                continue
            try:
                result = analyze_profile(profile_file, detected_gpu_count)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {profile_file}: {e}")

    print_analysis(results, available_vram, show_per_gpu=args.per_gpu)

    if args.suggest:
        print("\nOptimization Suggestions:")
        print("=" * 80)
        for result in results:
            if result['estimated_required_gb'] > available_vram:
                suggestions = suggest_optimizations(result, available_vram)
                if suggestions:
                    print(f"\n{result['profile_name']}:")
                    for suggestion in suggestions:
                        print(f"  • {suggestion}")

    if args.verbose:
        print("\nNotes:")
        print("- Model: Base model memory (distributed across GPUs for tensor parallelism)")
        print("- KV: KV cache memory per GPU")
        print("- Act: Activation memory per GPU")
        print("- Runtime: Calibrated runtime memory needs (1.15x theoretical)")
        print("- Peak: Peak memory during loading (1.1x runtime)")
        print("- Status: ✓ OK (fits), ⚠ Tight (close to limit), ✗ OOM (out of memory)")
        print("- Calibration based on real vLLM measurements")

if __name__ == "__main__":
    main()