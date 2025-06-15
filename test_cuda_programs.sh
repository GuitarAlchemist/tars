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
