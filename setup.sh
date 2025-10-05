#!/bin/bash

# vLLM Setup Script for Ubuntu 24.04
# This script automates the installation and configuration of vLLM with dual GPU support

set -e  # Exit on any error

# Configuration - Install in current directory
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root. Please run as a regular user with sudo privileges."
fi

# Show installation directory
info "Installation directory: $INSTALL_DIR"

# Check Ubuntu version
check_ubuntu_version() {
    log "Checking Ubuntu version..."
    
    if ! grep -q "24.04" /etc/os-release; then
        warn "This script is designed for Ubuntu 24.04. Current version:"
        cat /etc/os-release | grep VERSION
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        info "Ubuntu 24.04 detected ✓"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check for NVIDIA GPUs
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. Please install NVIDIA drivers first."
    fi
    
    # Check GPU count and memory
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    info "Detected $GPU_COUNT GPU(s)"
    
    # Check individual GPU memory
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found. Please install Python 3.8+."
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "Python version: $PYTHON_VERSION"
    
    # Convert version to comparable format (e.g., 3.12 -> 312, 3.8 -> 38)
    PYTHON_VERSION_NUM=$(echo "$PYTHON_VERSION" | sed 's/\.//')
    if [[ $PYTHON_VERSION_NUM -lt 38 ]]; then
        error "Python 3.8+ required. Current version: $PYTHON_VERSION"
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        info "CUDA version: $CUDA_VERSION"
    else
        warn "CUDA compiler not found. Will install CUDA toolkit."
    fi
    
    # Check system RAM
    TOTAL_RAM=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    info "Total system RAM: ${TOTAL_RAM}GB"
    
    if [[ $TOTAL_RAM -lt 32 ]]; then
        warn "Recommended minimum RAM is 32GB. Current: ${TOTAL_RAM}GB"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Install system dependencies
install_system_dependencies() {
    log "Installing system dependencies..."
    
    sudo apt update
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        wget \
        curl \
        bc \
        nvidia-cuda-toolkit \
        libnvidia-ml-dev
    
    info "System dependencies installed ✓"
}

# Setup Python environment
setup_python_environment() {
    log "Setting up Python virtual environment..."

    cd "$INSTALL_DIR"

    # Remove existing virtual environment if it exists (in case of re-runs)
    if [[ -d "vllm-env" ]]; then
        warn "Existing virtual environment found. Removing and recreating..."
        rm -rf vllm-env
    fi

    # Create virtual environment
    python3 -m venv vllm-env

    # Activate the virtual environment
    source vllm-env/bin/activate

    # Verify pip is working before upgrading
    if ! python -m pip --version &>/dev/null; then
        error "pip is not working in the virtual environment. Please check your Python installation."
    fi

    # Upgrade pip using python -m pip to ensure we use the correct pip
    python -m pip install --upgrade pip setuptools wheel

    info "Python environment created at $INSTALL_DIR/vllm-env ✓"
}

# Install vLLM and dependencies
install_vllm() {
    log "Installing vLLM and dependencies..."

    cd "$INSTALL_DIR"
    source vllm-env/bin/activate

    # Detect CUDA version for appropriate PyTorch installation
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d'.' -f1,2)
        info "Installing for CUDA $CUDA_VERSION"

        if [[ "$CUDA_VERSION" == "11.8" ]]; then
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            python -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
        else
            # Default to CUDA 12.1+
            python -m pip install torch torchvision torchaudio
            python -m pip install vllm
        fi
    else
        # Install latest versions
        python -m pip install torch torchvision torchaudio
        python -m pip install vllm
    fi

    # Install additional dependencies
    python -m pip install \
        transformers \
        accelerate \
        huggingface-hub \
        fastapi \
        uvicorn \
        pydantic \
        requests

    # Install NVIDIA ML Python bindings
    python -m pip install nvidia-ml-py3

    info "vLLM and dependencies installed ✓"
}

# Verify configuration files
verify_config_files() {
    log "Verifying configuration files..."

    cd "$INSTALL_DIR"

    # Check if required Python files exist
    local required_files=("multi_gpu_config.py" "basic_inference.py" "api_server.py" "monitor_gpus.py" "estimate_vram.py" "predownload.py" "activate_vllm.sh")

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file missing: $file. Please ensure all files are present in the repository."
        fi
    done

    info "All required configuration files present ✓"
}

# This function is no longer needed - files are tracked in repo
create_config_files() {
    verify_config_files
}

# Verify installation
verify_installation() {
    log "Verifying installation..."

    cd "$INSTALL_DIR"
    source vllm-env/bin/activate

    # Test vLLM import
    if python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null; then
        info "vLLM import successful ✓"
    else
        error "vLLM import failed"
    fi

    # Test GPU configuration
    if python multi_gpu_config.py; then
        info "GPU configuration verified ✓"
    else
        warn "GPU configuration check failed"
    fi

    # Test basic functionality with a small model
    info "Testing basic inference (this may take a few minutes for first run)..."

    cat > test_inference.py << 'EOF'
from vllm import LLM, SamplingParams
import torch

try:
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    # Use a very small model for testing
    llm = LLM(
        model="gpt2",
        tensor_parallel_size=min(gpu_count, 2),
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="float16"
    )

    sampling_params = SamplingParams(max_tokens=50)
    outputs = llm.generate(["Hello world"], sampling_params)

    print("Test successful!")
    print(f"Output: {outputs[0].outputs[0].text}")

except Exception as e:
    print(f"Test failed: {e}")
    exit(1)
EOF

    if python test_inference.py; then
        info "Basic inference test passed ✓"
        info "Note: Any 'Engine core proc died' message above is normal - it occurs when the test completes"
    else
        warn "Basic inference test failed - this may be normal for some systems"
    fi

    rm test_inference.py
}


# Create usage instructions
create_usage_instructions() {
    log "Creating usage instructions..."

    cd "$INSTALL_DIR"

    cat > README.md << EOF
# vLLM Setup - Usage Instructions

## Quick Start

1. **Activate the environment:**
   \`\`\`bash
   source activate_vllm.sh
   \`\`\`

2. **Check GPU configuration:**
   \`\`\`bash
   python multi_gpu_config.py
   \`\`\`

3. **Estimate vRAM requirements:**
   \`\`\`bash
   python estimate_vram.py
   \`\`\`

4. **Test basic inference:**
   \`\`\`bash
   python basic_inference.py
   \`\`\`

5. **Start API server:**
   \`\`\`bash
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
   \`\`\`

6. **Monitor GPU usage:**
   \`\`\`bash
   python monitor_gpus.py
   \`\`\`

## Environment Location

This installation is located at: $INSTALL_DIR/

To reactivate from anywhere:
\`\`\`bash
source $INSTALL_DIR/activate_vllm.sh
\`\`\`

## API Usage

The server is OpenAI-compatible. Start with:
\`\`\`bash
python api_server.py
\`\`\`

Then test with curl:
\`\`\`bash
curl -X POST "http://localhost:8000/v1/chat/completions" \\
     -H "Content-Type: application/json" \\
     -d '{
       "model": "RedHatAI/gemma-3-27b-it-quantized.w4a16",
       "messages": [{"role": "user", "content": "Hello, how are you?"}],
       "max_tokens": 100
     }'
\`\`\`

## Troubleshooting

1. **Out of Memory**: Reduce \`gpu_memory_utilization\` or \`max_model_len\`
2. **Slow Loading**: Ensure models are cached in \`~/.cache/huggingface/\`
3. **CUDA Errors**: Check \`nvidia-smi\` and restart if needed
EOF

    info "Usage instructions created ✓"
}

# Main installation function
main() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "  vLLM Setup for Ubuntu 24.04"
    echo "  Local Installation"
    echo "========================================"
    echo -e "${NC}"

    check_ubuntu_version
    check_prerequisites
    install_system_dependencies
    setup_python_environment
    install_vllm
    verify_config_files
    verify_installation
    create_usage_instructions

    log "Installation completed successfully!"
    echo
    info "Installation location: $INSTALL_DIR"
    info "Next steps:"
    echo "1. source activate_vllm.sh"
    echo "2. python multi_gpu_config.py"
    echo "3. python estimate_vram.py"
    echo "4. python basic_inference.py"
    echo
    info "For detailed instructions, see: $INSTALL_DIR/README.md"
    echo
    warn "Note: First model download may take time depending on your internet connection."
}

# Run main function
main "$@"
