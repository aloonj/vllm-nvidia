# vLLM Setup - Usage Instructions

## Quick Start

1. **Activate the environment:**
   ```bash
   source activate_vllm.sh
   ```

2. **Check GPU configuration:**
   ```bash
   python multi_gpu_config.py
   ```

3. **Test basic inference:**
   ```bash
   python basic_inference.py
   ```

4. **Start API server:**
   ```bash
   # List available profiles
   python api_server.py --list-profiles

   # Use a profile (auto-detects GPU count)
   python api_server.py --profile qwen30b
   python api_server.py --profile gemma27b

   # Override profile settings
   python api_server.py --profile gemma27b --port 8080 --max-model-len 8192

   # Force specific GPU count (override auto-detection)
   python api_server.py --profile qwen30b --tensor-parallel-size 1

   # Use 'auto' to detect GPUs with manual config
   python api_server.py \
     --model "meta-llama/Llama-2-13b-hf" \
     --tensor-parallel-size auto \
     --gpu-memory-utilization 0.9 \
     --max-model-len 4096 \
     --dtype float16

   # Use custom profile file
   python api_server.py --profile ~/my-profiles/custom.yaml

   # Show all options
   python api_server.py --help
   ```

5. **Monitor GPU usage:**
   ```bash
   python monitor_gpus.py
   ```

## Environment Location

This installation is located at: /home/ajames/vllm-nvidia/

To reactivate from anywhere:
```bash
source /home/ajames/vllm-nvidia/activate_vllm.sh
```

## API Usage

The server is OpenAI-compatible. Start with:
```bash
# Profile required (no built-in defaults)
python api_server.py --profile qwen30b
```

Then test with curl:
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "RedHatAI/gemma-3-27b-it-quantized.w4a16",
       "messages": [{"role": "user", "content": "Hello, how are you?"}],
       "max_tokens": 100
     }'
```

## Custom Profiles

Create your own model profiles in `profiles/` directory as YAML files:

```yaml
name: my_model
description: My custom model configuration
model: path/to/model
tensor_parallel_size: 2
gpu_memory_utilization: 0.95
max_model_len: 16384
dtype: bfloat16
```

See `profiles/README.md` for detailed documentation.

## Troubleshooting

1. **Out of Memory**: Reduce `gpu_memory_utilization` or `max_model_len`
2. **Slow Loading**: Ensure models are cached in `~/.cache/huggingface/`
3. **CUDA Errors**: Check `nvidia-smi` and restart if needed
