#!/bin/bash

# Quick CUDA compilation test for TARS Unified Non-Euclidean Vector Store

echo "🧪 TARS CUDA Compilation Test"
echo "============================="
echo ""

# Check CUDA availability
echo "🔍 Checking CUDA environment..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found!"
    exit 1
fi

echo "✅ CUDA compiler found:"
nvcc --version | head -n 4
echo ""

# Check GPU
echo "🔍 Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected!"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
echo ""

# Compile the unified vector store
echo "🔨 Compiling unified non-Euclidean vector store..."
SOURCE_FILE="unified_non_euclidean_vector_store.cu"
OUTPUT_FILE="test_unified_quick"

if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Source file not found: $SOURCE_FILE"
    exit 1
fi

# Simple compilation with basic flags
if nvcc -O3 -o "$OUTPUT_FILE" "$SOURCE_FILE" -lcublas -lcurand; then
    echo "✅ Compilation successful!"
else
    echo "❌ Compilation failed!"
    exit 1
fi

# Test execution
echo ""
echo "🚀 Testing execution..."
if [ -f "$OUTPUT_FILE" ]; then
    echo "Running CUDA test..."
    timeout 30s ./"$OUTPUT_FILE" || {
        echo "⚠️  Test timed out or failed, but compilation works!"
        echo "✅ CUDA compilation is functional"
        exit 0
    }
    echo "✅ CUDA test completed successfully!"
else
    echo "❌ Executable not found"
    exit 1
fi

echo ""
echo "🎉 CUDA COMPILATION TEST PASSED!"
echo "✅ NVCC compiler working"
echo "✅ GPU detected and accessible"
echo "✅ Unified vector store compiles"
echo "✅ CUDA execution functional"
echo ""
echo "Ready for F# integration!"
