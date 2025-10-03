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
   # Default configuration (Gemma-27B)
   python api_server.py

   # Use preset configurations
   python api_server.py --preset gemma27b  # RedHat Gemma-27B quantized
   python api_server.py --preset gemma12b  # Google Gemma-12B
   python api_server.py --preset llama13b  # Meta Llama-13B

   # Custom configuration
   python api_server.py --model "your-model" --port 8080 --tensor-parallel-size 2

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
python api_server.py
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

## Troubleshooting

1. **Out of Memory**: Reduce `gpu_memory_utilization` or `max_model_len`
2. **Slow Loading**: Ensure models are cached in `~/.cache/huggingface/`
3. **CUDA Errors**: Check `nvidia-smi` and restart if needed
