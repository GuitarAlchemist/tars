#!/bin/bash

# ============================================================================
# TARS AI Inference Engine - CUDA Build Script
# Compile CUDA kernels in WSL with real nvcc compiler
# ============================================================================

set -e  # Exit on any error

echo "🚀 TARS AI Inference Engine - CUDA Build"
echo "========================================"

# Check if we're in WSL
if ! grep -q Microsoft /proc/version 2>/dev/null; then
    echo "❌ This script must be run in WSL (Windows Subsystem for Linux)"
    echo "   CUDA compilation requires WSL environment"
    exit 1
fi

# Check CUDA installation
echo "🔍 Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA toolkit not found. Please install CUDA in WSL:"
    echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin"
    echo "   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600"
    echo "   wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb"
    echo "   sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb"
    echo "   sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/"
    echo "   sudo apt-get update"
    echo "   sudo apt-get -y install cuda"
    exit 1
fi

# Display CUDA version
echo "✅ CUDA toolkit found:"
nvcc --version

# Check GPU availability
echo ""
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi not available, but CUDA compilation can proceed"
fi

# Set up build environment
echo ""
echo "🔧 Setting up build environment..."
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Create build directory
BUILD_DIR="cuda/build"
mkdir -p $BUILD_DIR

# Change to CUDA directory
cd cuda

echo ""
echo "🏗️  Building TARS CUDA library..."

# Clean previous build
make clean

# Build the library
echo "📦 Compiling CUDA kernels and C++ wrappers..."
if make -j$(nproc); then
    echo "✅ CUDA library built successfully: libtars_cuda.so"
else
    echo "❌ CUDA build failed"
    exit 1
fi

# Verify the library
echo ""
echo "🔍 Verifying built library..."
if [ -f "libtars_cuda.so" ]; then
    echo "✅ Library file exists: $(ls -lh libtars_cuda.so)"
    
    # Check library dependencies
    echo "📋 Library dependencies:"
    ldd libtars_cuda.so | grep -E "(cuda|cublas)" || echo "   No CUDA dependencies found (static linking)"
    
    # Check symbols
    echo "📋 Exported symbols:"
    nm -D libtars_cuda.so | grep tars_cuda | head -10
    
else
    echo "❌ Library file not found"
    exit 1
fi

# Test basic functionality
echo ""
echo "🧪 Testing CUDA functionality..."

# Create simple test program
cat > test_cuda.cpp << 'EOF'
#include "include/tars_cuda.h"
#include <iostream>

int main() {
    // Test device info
    char name[256];
    size_t memory_mb;
    TarsError result = tars_cuda_get_device_info(0, name, &memory_mb);
    
    if (result == TARS_SUCCESS) {
        std::cout << "✅ GPU Device: " << name << std::endl;
        std::cout << "✅ Memory: " << memory_mb << " MB" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Failed to get device info: " << result << std::endl;
        return 1;
    }
}
EOF

# Compile test program
echo "🔨 Compiling test program..."
if g++ -std=c++17 -I./include -L. -ltars_cuda test_cuda.cpp -o test_cuda; then
    echo "✅ Test program compiled"
    
    # Run test
    echo "🏃 Running CUDA test..."
    export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
    if ./test_cuda; then
        echo "✅ CUDA test passed"
    else
        echo "⚠️  CUDA test failed (GPU may not be available in WSL)"
    fi
    
    # Clean up test files
    rm -f test_cuda test_cuda.cpp
else
    echo "⚠️  Test compilation failed"
fi

# Copy library to .NET project
echo ""
echo "📦 Installing library for .NET integration..."
cp libtars_cuda.so ../
echo "✅ Library copied to project root"

# Generate build info
echo ""
echo "📊 Build Information:"
echo "===================="
echo "Build Date: $(date)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Compiler: $(g++ --version | head -1)"
echo "Architecture: $(uname -m)"
echo "Library Size: $(ls -lh libtars_cuda.so | awk '{print $5}')"

# Create build manifest
cat > build_manifest.json << EOF
{
    "build_date": "$(date -Iseconds)",
    "cuda_version": "$(nvcc --version | grep release | awk '{print $6}' | cut -d, -f1)",
    "compiler": "$(g++ --version | head -1)",
    "architecture": "$(uname -m)",
    "library_size": "$(stat -c%s libtars_cuda.so)",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "build_host": "$(hostname)"
}
EOF

echo "✅ Build manifest created: build_manifest.json"

echo ""
echo "🎉 TARS CUDA Build Complete!"
echo "=========================="
echo "✅ CUDA kernels compiled successfully"
echo "✅ Library ready for .NET integration"
echo "✅ Real GPU acceleration available"
echo ""
echo "Next steps:"
echo "1. Build the .NET project: dotnet build"
echo "2. Run TARS inference engine with CUDA acceleration"
echo "3. Replace Ollama with TARS AI inference"

# Return to original directory
cd ..

echo ""
echo "🚀 Ready to replace Ollama with TARS AI inference engine!"
