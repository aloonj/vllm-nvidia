#!/usr/bin/env python3
"""
Basic vLLM Inference Example
Tests model loading and text generation with multi-GPU support
"""

from vllm import LLM, SamplingParams
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test vLLM inference")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium",
                       help="Model to use for testing")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                       help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                       help="Maximum model context length")
    args = parser.parse_args()

    # Check GPU setup
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    gpu_count = torch.cuda.device_count()
    print(f"Using {gpu_count} GPU(s)")

    # Initialize vLLM with multi-GPU support
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=min(gpu_count, 2),  # Use available GPUs
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="float16",
        trust_remote_code=True
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    # Test prompts
    prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Explain the benefits of renewable energy."
    ]

    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()