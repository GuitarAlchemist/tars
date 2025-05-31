#!/bin/bash

# TARS Agentic CUDA Vector Store Build Script
echo "🚀 Building TARS Agentic CUDA Vector Store..."
echo "============================================="

# Check CUDA installation
echo "🔍 Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✅ NVCC found: $(nvcc --version | grep release)"
else
    echo "❌ NVCC not found. Please install CUDA toolkit."
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
if nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  No GPU detected, but continuing with compilation..."
fi

# Set compilation flags
CUDA_FLAGS="-O3 -arch=sm_60 -lcublas"
SOURCE_FILE="tars_agentic_vector_store.cu"
OUTPUT_FILE="tars_agentic_vector_store"

echo ""
echo "🔨 Compiling CUDA source..."
echo "   Source: $SOURCE_FILE"
echo "   Output: $OUTPUT_FILE"
echo "   Flags: $CUDA_FLAGS"

# Compile the enhanced CUDA implementation
nvcc $CUDA_FLAGS -o $OUTPUT_FILE $SOURCE_FILE

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    
    # Check if binary was created
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✅ Binary created: $OUTPUT_FILE"
        ls -lh $OUTPUT_FILE
        
        echo ""
        echo "🧪 Running quick test..."
        ./$OUTPUT_FILE
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 TARS Agentic CUDA Vector Store built successfully!"
            echo "✅ Ready for F# integration"
            echo "✅ Ready for agentic RAG deployment"
            echo ""
            echo "📋 Next steps:"
            echo "   1. Test F# integration: dotnet run --project TarsEngine.FSharp.Core"
            echo "   2. Run agentic RAG demo: dotnet run --project TarsCli -- agentic-rag-demo"
            echo "   3. Deploy to production environment"
        else
            echo "❌ Test execution failed"
            exit 1
        fi
    else
        echo "❌ Binary not found after compilation"
        exit 1
    fi
else
    echo "❌ Compilation failed"
    exit 1
fi

echo ""
echo "🔧 Build script completed"
