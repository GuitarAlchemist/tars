#!/bin/bash

# CUDA Compilation Script for WSL
# Compiles TARS CUDA vector store implementations

echo "🚀 TARS CUDA COMPILATION FOR WSL"
echo "================================="
echo "Compiling CUDA vector store implementations..."
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found!"
    echo "Please install CUDA toolkit in WSL:"
    echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb"
    echo "  sudo dpkg -i cuda-keyring_1.0-1_all.deb"
    echo "  sudo apt-get update"
    echo "  sudo apt-get -y install cuda"
    exit 1
fi

# Check CUDA version
echo "📋 CUDA Environment:"
nvcc --version
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  nvidia-smi not available - GPU may not be accessible"
    echo ""
fi

# Create output directory
mkdir -p cuda_binaries
cd cuda_binaries

# CUDA compilation flags
CUDA_FLAGS="-O3 -arch=sm_75 -lcublas -lcurand"
CUDA_DIR="../src/TarsEngine.FSharp.Core/VectorStore/CUDA"

echo "🔧 Compiling CUDA Programs:"
echo "=========================="

# Compile key CUDA programs
CUDA_PROGRAMS=(
    "unified_non_euclidean_vector_store.cu:unified_vector_store"
    "tars_agentic_vector_store.cu:agentic_vector_store"
    "simple_wsl.cu:simple_wsl_test"
    "wsl_cuda_test.cu:wsl_cuda_test"
    "cuda_benchmark.cu:cuda_benchmark"
    "real_cuda_gpu_test.cu:real_gpu_test"
)

SUCCESS_COUNT=0
TOTAL_COUNT=${#CUDA_PROGRAMS[@]}

for program_info in "${CUDA_PROGRAMS[@]}"; do
    IFS=':' read -r source_file output_name <<< "$program_info"
    source_path="$CUDA_DIR/$source_file"
    
    echo "📄 Compiling $source_file..."
    
    if [ -f "$source_path" ]; then
        if nvcc $CUDA_FLAGS "$source_path" -o "$output_name" 2>/dev/null; then
            echo "   ✅ Success: $output_name"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "   ❌ Failed: $source_file"
            echo "   Trying with basic flags..."
            if nvcc -O2 "$source_path" -o "$output_name" 2>/dev/null; then
                echo "   ✅ Success with basic flags: $output_name"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo "   ❌ Failed with basic flags"
            fi
        fi
    else
        echo "   ❌ Source file not found: $source_path"
    fi
    echo ""
done

echo "📊 Compilation Summary:"
echo "======================"
echo "Successful: $SUCCESS_COUNT/$TOTAL_COUNT"
echo ""

# List compiled binaries
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "🎯 Compiled Binaries:"
    echo "===================="
    ls -la *.exe 2>/dev/null || ls -la * 2>/dev/null | grep -v "\.cu$"
    echo ""
    
    echo "🚀 Running Tests:"
    echo "================"
    
    # Test simple WSL program first
    if [ -f "simple_wsl_test" ]; then
        echo "🧪 Testing simple WSL CUDA program..."
        ./simple_wsl_test
        echo ""
    fi
    
    # Test unified vector store
    if [ -f "unified_vector_store" ]; then
        echo "🧪 Testing unified non-Euclidean vector store..."
        ./unified_vector_store
        echo ""
    fi
    
    # Test agentic vector store
    if [ -f "agentic_vector_store" ]; then
        echo "🧪 Testing agentic vector store..."
        ./agentic_vector_store
        echo ""
    fi
    
    # Run benchmark if available
    if [ -f "cuda_benchmark" ]; then
        echo "🧪 Running CUDA benchmark..."
        ./cuda_benchmark
        echo ""
    fi
    
    # Test real GPU functionality
    if [ -f "real_gpu_test" ]; then
        echo "🧪 Testing real GPU functionality..."
        ./real_gpu_test
        echo ""
    fi
    
else
    echo "❌ No programs compiled successfully"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "==================="
    echo "1. Check CUDA installation: nvcc --version"
    echo "2. Check GPU driver: nvidia-smi"
    echo "3. Verify WSL2 with GPU support is enabled"
    echo "4. Install CUDA toolkit for WSL"
    echo ""
fi

echo "🎉 CUDA Compilation Complete!"
echo "============================="

# Return to original directory
cd ..

# Create a simple test script
cat > test_cuda_programs.sh << 'EOF'
#!/bin/bash
echo "🧪 TARS CUDA Test Suite"
echo "======================"
cd cuda_binaries

for program in unified_vector_store agentic_vector_store simple_wsl_test; do
    if [ -f "$program" ]; then
        echo "Testing $program..."
        ./"$program"
        echo "---"
    fi
done
EOF

chmod +x test_cuda_programs.sh

echo "📝 Created test_cuda_programs.sh for easy testing"
echo "Run: ./test_cuda_programs.sh"
