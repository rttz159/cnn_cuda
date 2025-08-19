#!/bin/bash
set -e  # Exit immediately if a command fails

echo "=== Installing build tools and OpenCV ==="
sudo apt update
sudo apt install -y build-essential libopencv-dev ninja-build

echo "=== Adding NVIDIA CUDA repo key ==="
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

echo "=== Updating package list ==="
sudo apt-get update

echo "=== Installing CUDA Toolkit 13.0 ==="
sudo apt-get -y install cuda-toolkit-13-0

echo "=== Updating environment variables ==="
if ! grep -q "cuda-13.0/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

echo "=== Reloading ~/.bashrc ==="
source ~/.bashrc

echo "=== Installation complete! ==="
nvcc --version || echo "CUDA installation might need re-login"
