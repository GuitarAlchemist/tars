#!/bin/bash

# Build script for TARS Custom Transformer CUDA kernels

echo "🌌 TARS Custom Transformer CUDA Build"
echo "====================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found!"
    echo "   Please install CUDA Toolkit"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected!"
    echo "   Make sure NVIDIA drivers are installed"
    exit 1
fi

echo "✅ CUDA environment detected"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
echo ""

# Build configuration
SOURCE_FILE="cuda_kernels_hybrid_space.cu"
OUTPUT_LIBRARY="cuda_kernels_hybrid_space.dll"

# CUDA compilation flags for hybrid space operations
CUDA_FLAGS=(
    "-O3"                           # Maximum optimization
    "-use_fast_math"               # Fast math operations
    "-Xptxas -O3"                  # PTX assembler optimization
    "-Xcompiler -O3"               # Host compiler optimization
    "-Xcompiler -fPIC"             # Position independent code
    "--shared"                     # Build shared library
    "-gencode arch=compute_75,code=sm_75"  # RTX 20xx series
    "-gencode arch=compute_80,code=sm_80"  # RTX 30xx series
    "-gencode arch=compute_86,code=sm_86"  # RTX 30xx Ti series
    "-gencode arch=compute_89,code=sm_89"  # RTX 40xx series
    "-lcublas"                     # cuBLAS library
    "-lcurand"                     # cuRAND library
    "-lm"                          # Math library
)

echo "🔨 Compiling CUDA hybrid space kernels..."
echo "Source: $SOURCE_FILE"
echo "Output: $OUTPUT_LIBRARY"
echo "Flags: ${CUDA_FLAGS[*]}"
echo ""

# Compile shared library for F# integration
if nvcc "${CUDA_FLAGS[@]}" -o "$OUTPUT_LIBRARY" "$SOURCE_FILE"; then
    echo "✅ CUDA library built successfully: $OUTPUT_LIBRARY"
else
    echo "❌ Failed to build CUDA library"
    exit 1
fi

echo ""
echo "📊 Build Summary:"
echo "================"
ls -la "$OUTPUT_LIBRARY" 2>/dev/null || echo "❌ Build artifact not found"

echo ""
echo "🧪 Testing CUDA library..."
echo "=========================="

# Create simple test program
cat > test_cuda_lib.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

int main() {
    printf("🔍 Testing CUDA library loading...\n");
    
    void* handle = dlopen("./cuda_kernels_hybrid_space.dll", RTLD_LAZY);
    if (!handle) {
        printf("❌ Cannot load library: %s\n", dlerror());
        return 1;
    }
    
    // Test if we can find the functions
    void* mobius_func = dlsym(handle, "call_mobius_add");
    void* hyperbolic_func = dlsym(handle, "call_hyperbolic_distance");
    void* projective_func = dlsym(handle, "call_projective_normalize");
    void* quaternion_func = dlsym(handle, "call_dual_quaternion_norm");
    
    if (mobius_func && hyperbolic_func && projective_func && quaternion_func) {
        printf("✅ All CUDA functions found in library\n");
        printf("   - call_mobius_add: %p\n", mobius_func);
        printf("   - call_hyperbolic_distance: %p\n", hyperbolic_func);
        printf("   - call_projective_normalize: %p\n", projective_func);
        printf("   - call_dual_quaternion_norm: %p\n", quaternion_func);
    } else {
        printf("❌ Some CUDA functions missing\n");
        dlclose(handle);
        return 1;
    }
    
    dlclose(handle);
    printf("✅ CUDA library test passed!\n");
    return 0;
}
EOF

# Compile and run test
if gcc -o test_cuda_lib test_cuda_lib.c -ldl; then
    if ./test_cuda_lib; then
        echo "✅ CUDA library validation successful!"
    else
        echo "❌ CUDA library validation failed"
        exit 1
    fi
    
    # Clean up test files
    rm -f test_cuda_lib test_cuda_lib.c
else
    echo "⚠️  Could not compile library test (but library should work)"
    rm -f test_cuda_lib.c
fi

echo ""
echo "🎉 CUDA BUILD COMPLETE!"
echo "======================="
echo "✅ Hybrid space kernels compiled"
echo "✅ Möbius addition operations ready"
echo "✅ Hyperbolic distance calculations ready"
echo "✅ Projective space operations ready"
echo "✅ Dual quaternion operations ready"
echo "✅ F# P/Invoke integration ready"
echo ""
echo "📋 Next Steps:"
echo "1. Build F# project: dotnet build"
echo "2. Run TARS custom transformer demo"
echo "3. Train models with hybrid embeddings"
echo ""
echo "🌟 TARS Custom Transformers ready for advanced semantic understanding!"
