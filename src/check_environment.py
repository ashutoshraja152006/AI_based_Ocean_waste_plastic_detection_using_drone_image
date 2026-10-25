import os
import sys
import platform
import subprocess


def check_system_info():
    print("=" * 60)
    print("1. SYSTEM & PYTHON ENVIRONMENT")
    print("=" * 60)
    print(f"OS / Platform  : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture   : {platform.machine()}")
    print(f"Processor      : {platform.processor()}")
    print(f"CPU Cores      : {os.cpu_count()} logical cores")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Python Path    : {sys.executable}")

    # Check total RAM on Windows using wmic / powershell
    try:
        cmd = ["powershell", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]
        out = subprocess.check_output(cmd, text=True).strip()
        ram_gb = int(out) / (1024 ** 3)
        print(f"Total System RAM: {ram_gb:.2f} GB")
    except Exception as e:
        print(f"Total System RAM: Could not detect automatically ({e})")


def check_nvidia_gpu():
    print("\n" + "=" * 60)
    print("2. NVIDIA GPU & CUDA DRIVER CHECK (nvidia-smi)")
    print("=" * 60)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        print(f"NVIDIA GPU Found : YES")
        for idx, line in enumerate(out.splitlines()):
            print(f"  GPU {idx}: {line}")
    except FileNotFoundError:
        print("NVIDIA GPU Found : NO ('nvidia-smi' command not found - likely integrated graphics/CPU only)")
    except Exception as e:
        print(f"NVIDIA GPU Check : Returned error ({e})")


def check_pytorch():
    print("\n" + "=" * 60)
    print("3. PYTORCH & GPU AVAILABILITY IN PYTHON")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch Installed       : YES (Version: {torch.__version__})")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available in Torch : {cuda_available}")
        if cuda_available:
            print(f"CUDA Version (Torch)    : {torch.version.cuda}")
            print(f"GPU Device Name         : {torch.cuda.get_device_name(0)}")
        else:
            print("Note: PyTorch is currently running in CPU-only mode.")
    except ImportError:
        print("PyTorch Installed       : NO (Not installed in this Python environment yet)")
    print("=" * 60)


if __name__ == "__main__":
    check_system_info()
    check_nvidia_gpu()
    check_pytorch()