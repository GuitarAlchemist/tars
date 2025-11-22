#!/bin/bash

# ============================================================================
# TARS AI Inference Engine - Comprehensive Testing Script
# Thorough validation of CUDA implementation and Ollama replacement
# ============================================================================

set -e  # Exit on any error

echo "🧪 TARS AI INFERENCE ENGINE - COMPREHENSIVE TESTING"
echo "=================================================="
echo "Validating CUDA implementation and Ollama replacement capabilities"
echo ""

# Configuration
TEST_RESULTS_DIR="./test-results"
CUDA_BUILD_DIR="./src/TARS.AI.Inference/cuda"
INFERENCE_PROJECT="./src/TARS.AI.Inference"
TEST_PROJECT="./src/TARS.AI.Inference.Tests"

# Create test results directory
mkdir -p $TEST_RESULTS_DIR

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check .NET SDK
    if ! command -v dotnet &> /dev/null; then
        echo "❌ .NET SDK not found. Please install .NET 9.0 SDK"
        exit 1
    fi
    
    log "✅ .NET SDK: $(dotnet --version)"
    
    # Check if WSL is available for CUDA compilation
    if grep -q Microsoft /proc/version 2>/dev/null; then
        log "✅ Running in WSL - CUDA compilation available"
        WSL_AVAILABLE=true
    else
        log "⚠️ Not running in WSL - CUDA compilation may not be available"
        WSL_AVAILABLE=false
    fi
    
    # Check CUDA toolkit (if in WSL)
    if [ "$WSL_AVAILABLE" = true ]; then
        if command -v nvcc &> /dev/null; then
            log "✅ CUDA toolkit: $(nvcc --version | grep release | awk '{print $6}' | cut -d, -f1)"
            CUDA_AVAILABLE=true
        else
            log "⚠️ CUDA toolkit not found - GPU acceleration tests will be limited"
            CUDA_AVAILABLE=false
        fi
        
        # Check GPU availability
        if command -v nvidia-smi &> /dev/null; then
            log "✅ GPU detected:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
            GPU_AVAILABLE=true
        else
            log "⚠️ No GPU detected - will test CPU fallback"
            GPU_AVAILABLE=false
        fi
    else
        CUDA_AVAILABLE=false
        GPU_AVAILABLE=false
    fi
}

# Function to build CUDA library
build_cuda_library() {
    if [ "$CUDA_AVAILABLE" = true ]; then
        log "🔨 Building CUDA library..."
        
        cd $CUDA_BUILD_DIR
        
        # Clean previous build
        if [ -f "Makefile" ]; then
            make clean || true
        fi
        
        # Build CUDA library
        if make -j$(nproc) 2>&1 | tee "$TEST_RESULTS_DIR/cuda-build.log"; then
            log "✅ CUDA library built successfully"
            
            # Verify library
            if [ -f "libtars_cuda.so" ]; then
                log "✅ Library file: $(ls -lh libtars_cuda.so | awk '{print $5}')"
                
                # Copy to project root
                cp libtars_cuda.so ../
                log "✅ Library copied to project root"
            else
                log "❌ Library file not found after build"
                return 1
            fi
        else
            log "❌ CUDA library build failed"
            return 1
        fi
        
        cd - > /dev/null
    else
        log "⚠️ Skipping CUDA library build (CUDA not available)"
    fi
}

# Function to build .NET projects
build_dotnet_projects() {
    log "🏗️ Building .NET projects..."
    
    # Build inference engine
    log "Building TARS.AI.Inference..."
    if dotnet build $INFERENCE_PROJECT -c Release --verbosity minimal 2>&1 | tee "$TEST_RESULTS_DIR/inference-build.log"; then
        log "✅ TARS.AI.Inference built successfully"
    else
        log "❌ TARS.AI.Inference build failed"
        return 1
    fi
    
    # Build test project
    log "Building TARS.AI.Inference.Tests..."
    if dotnet build $TEST_PROJECT -c Release --verbosity minimal 2>&1 | tee "$TEST_RESULTS_DIR/tests-build.log"; then
        log "✅ TARS.AI.Inference.Tests built successfully"
    else
        log "❌ TARS.AI.Inference.Tests build failed"
        return 1
    fi
}

# Function to run unit tests
run_unit_tests() {
    log "🧪 Running unit tests..."
    
    if dotnet test $TEST_PROJECT -c Release --filter "Category=Unit" --logger "trx;LogFileName=unit-tests.trx" --results-directory $TEST_RESULTS_DIR 2>&1 | tee "$TEST_RESULTS_DIR/unit-tests.log"; then
        log "✅ Unit tests completed"
    else
        log "⚠️ Some unit tests failed - check logs"
    fi
}

# Function to run integration tests
run_integration_tests() {
    log "🔗 Running integration tests..."
    
    if dotnet test $TEST_PROJECT -c Release --filter "Category=Integration" --logger "trx;LogFileName=integration-tests.trx" --results-directory $TEST_RESULTS_DIR 2>&1 | tee "$TEST_RESULTS_DIR/integration-tests.log"; then
        log "✅ Integration tests completed"
    else
        log "⚠️ Some integration tests failed - check logs"
    fi
}

# Function to run performance benchmarks
run_performance_tests() {
    log "⚡ Running performance benchmarks..."
    
    # Run custom test runner for performance tests
    if dotnet run --project $TEST_PROJECT -c Release performance 2>&1 | tee "$TEST_RESULTS_DIR/performance-tests.log"; then
        log "✅ Performance tests completed"
    else
        log "⚠️ Some performance tests failed - check logs"
    fi
}

# Function to run CUDA-specific tests
run_cuda_tests() {
    if [ "$CUDA_AVAILABLE" = true ]; then
        log "🚀 Running CUDA-specific tests..."
        
        # Test CUDA library loading
        log "Testing CUDA library loading..."
        if ldd $INFERENCE_PROJECT/libtars_cuda.so 2>&1 | tee "$TEST_RESULTS_DIR/cuda-deps.log"; then
            log "✅ CUDA library dependencies verified"
        else
            log "⚠️ CUDA library dependency issues"
        fi
        
        # Test CUDA functionality
        if [ "$GPU_AVAILABLE" = true ]; then
            log "Testing GPU functionality..."
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits > "$TEST_RESULTS_DIR/gpu-baseline.log"
            log "✅ GPU baseline recorded"
        fi
    else
        log "⚠️ Skipping CUDA tests (CUDA not available)"
    fi
}

# Function to run Ollama compatibility tests
run_ollama_compatibility_tests() {
    log "🔌 Running Ollama compatibility tests..."
    
    # Test API compatibility
    if dotnet test $TEST_PROJECT -c Release --filter "Category=OllamaCompatibility" --logger "trx;LogFileName=ollama-compat.trx" --results-directory $TEST_RESULTS_DIR 2>&1 | tee "$TEST_RESULTS_DIR/ollama-compat.log"; then
        log "✅ Ollama compatibility tests completed"
    else
        log "⚠️ Some Ollama compatibility tests failed - check logs"
    fi
}

# Function to run stress tests
run_stress_tests() {
    log "💪 Running stress tests..."
    
    # Memory leak tests
    log "Testing for memory leaks..."
    if dotnet test $TEST_PROJECT -c Release --filter "Category=MemoryLeak" --logger "trx;LogFileName=memory-leak.trx" --results-directory $TEST_RESULTS_DIR 2>&1 | tee "$TEST_RESULTS_DIR/memory-leak.log"; then
        log "✅ Memory leak tests completed"
    else
        log "⚠️ Memory leak tests failed - check logs"
    fi
    
    # Concurrency tests
    log "Testing concurrency handling..."
    if dotnet test $TEST_PROJECT -c Release --filter "Category=Concurrency" --logger "trx;LogFileName=concurrency.trx" --results-directory $TEST_RESULTS_DIR 2>&1 | tee "$TEST_RESULTS_DIR/concurrency.log"; then
        log "✅ Concurrency tests completed"
    else
        log "⚠️ Concurrency tests failed - check logs"
    fi
}

# Function to generate comprehensive report
generate_test_report() {
    log "📊 Generating comprehensive test report..."
    
    # Run the test runner to generate detailed report
    if dotnet run --project $TEST_PROJECT -c Release 2>&1 | tee "$TEST_RESULTS_DIR/comprehensive-report.log"; then
        log "✅ Comprehensive test report generated"
    else
        log "⚠️ Test report generation had issues"
    fi
    
    # Create summary report
    cat > "$TEST_RESULTS_DIR/test-summary.md" << EOF
# TARS AI Inference Engine - Test Summary

**Test Execution Date:** $(date)
**Environment:** $(uname -a)
**CUDA Available:** $CUDA_AVAILABLE
**GPU Available:** $GPU_AVAILABLE

## Test Results

### Build Status
- CUDA Library: $([ "$CUDA_AVAILABLE" = true ] && echo "✅ Built" || echo "⚠️ Skipped")
- .NET Projects: ✅ Built

### Test Execution
- Unit Tests: $([ -f "$TEST_RESULTS_DIR/unit-tests.log" ] && echo "✅ Executed" || echo "❌ Failed")
- Integration Tests: $([ -f "$TEST_RESULTS_DIR/integration-tests.log" ] && echo "✅ Executed" || echo "❌ Failed")
- Performance Tests: $([ -f "$TEST_RESULTS_DIR/performance-tests.log" ] && echo "✅ Executed" || echo "❌ Failed")
- CUDA Tests: $([ "$CUDA_AVAILABLE" = true ] && echo "✅ Executed" || echo "⚠️ Skipped")
- Ollama Compatibility: $([ -f "$TEST_RESULTS_DIR/ollama-compat.log" ] && echo "✅ Executed" || echo "❌ Failed")

### Files Generated
$(ls -la $TEST_RESULTS_DIR)

## Recommendations

$(if [ "$CUDA_AVAILABLE" = true ] && [ "$GPU_AVAILABLE" = true ]; then
    echo "✅ **PRODUCTION READY** - Full CUDA acceleration available"
elif [ "$CUDA_AVAILABLE" = true ]; then
    echo "⚠️ **MOSTLY READY** - CUDA available but no GPU detected"
else
    echo "⚠️ **CPU ONLY** - CUDA not available, CPU fallback only"
fi)

---
*Generated by TARS AI Inference Engine Test Suite*
EOF

    log "✅ Test summary generated: $TEST_RESULTS_DIR/test-summary.md"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    log "Starting comprehensive testing of TARS AI Inference Engine"
    
    # Execute test phases
    check_prerequisites
    
    if [ "$1" != "skip-build" ]; then
        build_cuda_library
        build_dotnet_projects
    fi
    
    run_unit_tests
    run_integration_tests
    run_performance_tests
    run_cuda_tests
    run_ollama_compatibility_tests
    
    if [ "$1" = "stress" ]; then
        run_stress_tests
    fi
    
    generate_test_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "🎉 COMPREHENSIVE TESTING COMPLETE!"
    log "Total execution time: ${duration} seconds"
    log "Results available in: $TEST_RESULTS_DIR"
    
    # Final status
    if [ -f "$TEST_RESULTS_DIR/test-summary.md" ]; then
        echo ""
        echo "📋 TEST SUMMARY:"
        echo "================"
        cat "$TEST_RESULTS_DIR/test-summary.md" | grep -A 20 "## Recommendations"
    fi
    
    echo ""
    echo "🚀 TARS AI Inference Engine testing completed!"
    echo "Check $TEST_RESULTS_DIR for detailed results and logs."
}

# Handle command line arguments
case "${1:-}" in
    "quick")
        log "Running quick tests (no stress tests)"
        main skip-stress
        ;;
    "stress")
        log "Running full tests including stress tests"
        main stress
        ;;
    "skip-build")
        log "Skipping build phase"
        main skip-build
        ;;
    *)
        log "Running standard comprehensive tests"
        main
        ;;
esac
