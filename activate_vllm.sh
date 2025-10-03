#!/bin/bash
# vLLM Environment Activation Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/vllm-env" ]; then
    echo "Error: vLLM environment not found at $SCRIPT_DIR/vllm-env"
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

source "$SCRIPT_DIR/vllm-env/bin/activate"

echo "vLLM environment activated!"
echo "Location: $SCRIPT_DIR"
echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'Not installed')"

# Set CUDA environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_MODELSCOPE=False

echo ""
echo "Available commands:"
echo "  python multi_gpu_config.py           - Check GPU configuration"
echo "  python basic_inference.py            - Test basic inference"
echo "  python api_server.py                 - Start OpenAI-compatible API server"
echo "  python api_server.py --help          - Show all server options"
echo "  python api_server.py --preset gemma27b - Use Gemma-27B preset"
echo "  python monitor_gpus.py               - Monitor GPU usage"
echo ""