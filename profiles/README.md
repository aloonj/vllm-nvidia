# vLLM Server Profiles

This directory contains YAML configuration profiles for the vLLM API server. Profiles allow you to save and reuse model configurations without modifying the main script.

## Using Profiles

### List available profiles:
```bash
python api_server_v2.py --list-profiles
```

### Use a profile:
```bash
python api_server_v2.py --profile qwen30b
```

### Use a custom profile file:
```bash
python api_server_v2.py --profile /path/to/custom.yaml
```

### Override profile settings:
```bash
python api_server_v2.py --profile gemma27b --port 8080 --max-model-len 8192
```

## Profile Format

Profiles are YAML files with the following structure:

```yaml
# Profile metadata (optional)
name: profile_name
description: Description of the profile

# Required fields
model: model/name-or-path
tensor_parallel_size: auto  # Use 'auto' to detect GPUs, or specify a number
gpu_memory_utilization: 0.9
max_model_len: 16384
dtype: bfloat16

# Optional fields
quantization: gptq
trust_remote_code: true
enable_prefix_caching: true
enable_chunked_prefill: true
chunked_prefill_size: 8192
max_num_batched_tokens: 65536
max_num_seqs: 4
kv_cache_dtype: fp8
port: 8000
host: 0.0.0.0
```

## Profile Locations

The server searches for profiles in these directories (in order):
1. `./profiles/` - Current directory
2. `~/.vllm/profiles/` - User home directory
3. Script directory's `profiles/` folder

## Creating Custom Profiles

1. Copy an existing profile as a template
2. Modify the settings for your model and hardware
3. Save with a descriptive name (e.g., `my_model.yaml`)
4. Test with `python api_server_v2.py --profile my_model`

## Example Profiles

- `gemma27b.yaml` - RedHat Gemma 3 27B quantized model
- `qwen30b.yaml` - Qwen3 30B GPTQ optimized for dual 3090s
- `llama13b.yaml` - Standard Llama 2 13B configuration
- `gemma12b.yaml` - Gemma 2 12B model configuration

## Auto GPU Detection

The `tensor_parallel_size` parameter supports automatic GPU detection:

- **`auto`**: Automatically detects and uses all available GPUs
- **Number**: Use specific number of GPUs (e.g., `1`, `2`, `4`)

Examples:
```yaml
# Auto-detect GPUs (recommended for portability)
tensor_parallel_size: auto

# Fixed GPU count (when you need specific configuration)
tensor_parallel_size: 2
```

## Tips

- Use `tensor_parallel_size: auto` for profiles shared across different systems
- Use descriptive names for custom profiles
- Include comments in YAML to document special settings
- Test profiles with different batch sizes for optimal performance
- Share profiles with your team via version control