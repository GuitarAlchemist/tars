#!/bin/bash

# TARS Unified Non-Euclidean CUDA Vector Store Build Script
# Optimized for WSL with CUDA support

echo "🌌 TARS Unified Non-Euclidean CUDA Vector Store Build"
echo "===================================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found!"
    echo "   Please install CUDA Toolkit on WSL"
    echo "   Guide: https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected!"
    echo "   Make sure NVIDIA drivers are installed and WSL has GPU access"
    exit 1
fi

echo "✅ CUDA environment detected"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
echo ""

# Build configuration
SOURCE_FILE="unified_non_euclidean_vector_store.cu"
OUTPUT_EXECUTABLE="unified_non_euclidean_vector_store"
OUTPUT_LIBRARY="libunified_non_euclidean_vector_store.so"

# CUDA compilation flags for advanced optimization
CUDA_FLAGS=(
    "-O3"                           # Maximum optimization
    "-use_fast_math"               # Fast math operations
    "-Xptxas -O3"                  # PTX assembler optimization
    "-Xcompiler -O3"               # Host compiler optimization
    "-Xcompiler -fPIC"             # Position independent code for shared library
    "-gencode arch=compute_75,code=sm_75"  # RTX 20xx series
    "-gencode arch=compute_80,code=sm_80"  # RTX 30xx series
    "-gencode arch=compute_86,code=sm_86"  # RTX 30xx Ti series
    "-gencode arch=compute_89,code=sm_89"  # RTX 40xx series
    "-lcublas"                     # cuBLAS library
    "-lcurand"                     # cuRAND library
    "-lm"                          # Math library
)

echo "🔨 Compiling CUDA source..."
echo "Source: $SOURCE_FILE"
echo "Flags: ${CUDA_FLAGS[*]}"
echo ""

# Compile executable for testing
echo "📦 Building executable..."
if nvcc "${CUDA_FLAGS[@]}" -o "$OUTPUT_EXECUTABLE" "$SOURCE_FILE"; then
    echo "✅ Executable built successfully: $OUTPUT_EXECUTABLE"
else
    echo "❌ Failed to build executable"
    exit 1
fi

# Compile shared library for F# integration
echo "📚 Building shared library..."
LIBRARY_FLAGS=("${CUDA_FLAGS[@]}" "--shared")
if nvcc "${LIBRARY_FLAGS[@]}" -o "$OUTPUT_LIBRARY" "$SOURCE_FILE"; then
    echo "✅ Shared library built successfully: $OUTPUT_LIBRARY"
else
    echo "❌ Failed to build shared library"
    exit 1
fi

echo ""
echo "📊 Build Summary:"
echo "================"
ls -la "$OUTPUT_EXECUTABLE" "$OUTPUT_LIBRARY" 2>/dev/null || echo "❌ Build artifacts not found"

echo ""
echo "🧪 Running CUDA functionality test..."
echo "====================================="

# Test the executable
if [ -f "$OUTPUT_EXECUTABLE" ]; then
    echo "🚀 Testing unified non-Euclidean vector store..."
    ./"$OUTPUT_EXECUTABLE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 CUDA BUILD AND TEST SUCCESSFUL!"
        echo "✅ Unified Non-Euclidean Vector Store is ready"
        echo "✅ Multiple geometric spaces supported"
        echo "✅ CUDA acceleration verified"
        echo "✅ F# integration library available"
        echo ""
        echo "📋 Next Steps:"
        echo "1. Copy $OUTPUT_LIBRARY to F# project directory"
        echo "2. Update F# project to reference the library"
        echo "3. Test F# bindings with real data"
        echo ""
        echo "🌟 TARS now has advanced non-Euclidean semantic understanding!"
    else
        echo "❌ CUDA test failed"
        exit 1
    fi
else
    echo "❌ Executable not found"
    exit 1
fi

# Optional: Copy library to F# project if path is provided
if [ ! -z "$1" ]; then
    FSHARP_PROJECT_PATH="$1"
    echo "📁 Copying library to F# project: $FSHARP_PROJECT_PATH"
    
    if [ -d "$FSHARP_PROJECT_PATH" ]; then
        cp "$OUTPUT_LIBRARY" "$FSHARP_PROJECT_PATH/"
        echo "✅ Library copied to F# project"
    else
        echo "❌ F# project path not found: $FSHARP_PROJECT_PATH"
    fi
fi

echo ""
echo "🌌 TARS Unified Non-Euclidean CUDA Vector Store Build Complete!"
echo "Ready for advanced semantic operations in multiple geometric spaces!"
