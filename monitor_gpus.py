#!/usr/bin/env python3
"""
GPU Monitoring Tool for vLLM
Real-time monitoring of GPU usage, memory, temperature, and power
"""

import time
import argparse
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

# Import NVIDIA ML library at module level
nvml_available = False
device_count = 0

try:
    from pynvml import *
    nvmlInit()
    nvml_available = True
    device_count = nvmlDeviceGetCount()
except ImportError:
    print("Error: NVIDIA ML library not available.")
    print("Install with: pip install nvidia-ml-py3")
    exit(1)
except Exception as e:
    print(f"Error: NVIDIA ML initialization failed: {e}")
    exit(1)

def monitor_gpus(interval=5, duration=None):
    """Monitor GPU usage and memory"""

    start_time = time.time()

    try:
        while True:
            current_time = time.time()

            # Check duration limit
            if duration and (current_time - start_time) > duration:
                break

            print("\n" + "="*80)
            print(f"GPU Monitoring - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)

            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)

                # Device name
                name = nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                # Memory info
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem_info.used / 1024**3
                mem_total_gb = mem_info.total / 1024**3
                mem_percent = (mem_info.used / mem_info.total) * 100

                # GPU utilization
                util = nvmlDeviceGetUtilizationRates(handle)

                # Temperature
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

                # Power
                try:
                    power = nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    power_str = f"{power:.1f}W"
                except:
                    power_str = "N/A"

                print(f"GPU {i} ({name}):")
                print(f"  Memory: {mem_used_gb:.2f}GB/{mem_total_gb:.2f}GB ({mem_percent:.1f}%)")
                print(f"  Utilization: {util.gpu}%")
                print(f"  Temperature: {temp}°C")
                print(f"  Power: {power_str}")
                print()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if nvml_available:
            nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU usage")
    parser.add_argument("--interval", "-i", type=int, default=5,
                       help="Update interval in seconds (default: 5)")
    parser.add_argument("--duration", "-d", type=int,
                       help="Duration in seconds (unlimited if not specified)")

    args = parser.parse_args()
    monitor_gpus(args.interval, args.duration)