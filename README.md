# PyTorch for TensorTorrent Hardware: Installation Guide

> **Note**: This is a very early attempt to port of PyTorch (eager) to TensorTorrent hardware. Feel free to leave feedback and open PRs to cover more features.
> Right now, the exact functionality that works is covered by (yep, even the dimensions are hardcoded to the ones from `eltwise_binary.cpp`):
> ```python
> import torch
> a = torch.ones(2 * 1024 * 64)
> b = a.to("tt")
> c = b + b
> c.to("cpu")[:2 * 2 * 1024]
> ```

## Setup Instructions

### 1. TT-Metal Setup
```bash
# Clone the TensorTorrent Metal repository
git clone https://github.com/tenstorrent/tt-metal
cd tt-metal

# Initialize and update submodules
git submodule update --init --recursive

# Set environment variable and build
export TT_METAL_HOME=/root/tt-metal
./build_metal.sh --build-programming-examples ./build/programming_examples/eltwise_binary
```

In addition you currently need to clone https://github.com/qlibs/reflect into `/root/reflect`.

### 2. Miniforge Installation
```bash
# Download and install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

### 3. PyTorch Installation
```bash
# Clone the TensorTorrent-compatible PyTorch fork
git clone https://github.com/pcmoritz/pytorch
cd pytorch

# Switch to the TensorTorrent branch
git checkout tt

# Install requirements
pip install -r requirements.txt

# Set needed environment variables to use clang and libc++
export USE_MKLDNN=OFF
export CC=/usr/bin/clang-17
export CXX=/usr/bin/clang++-17
export CXXFLAGS=-stdlib=libc++

# Build PyTorch
python setup.py develop
```

## Useful Resources

- [Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [Memory on TensorTorrent (Blog)](https://clehaxze.tw/gemlog/2025/03-17-memory-on-tenstorrent.gmi)
- [TensorTorrent Tutorial Video](https://www.youtube.com/watch?v=Fjyw5L5aQsQ)
- [George Hotz's TensorTorrent Twitch Project](https://github.com/geohot/tt-twitch)
