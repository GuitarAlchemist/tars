# TARS CUDA Vector Store Makefile
# Compile in WSL with CUDA support

# CUDA Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -shared -Xcompiler -fPIC -O3 -arch=sm_75
CUDA_LIBS = -lcublas -lcurand

# Source and target files
SOURCE = cuda_vector_store.cu
TARGET = libtars_cuda.so
TEST_TARGET = tars_cuda_test

# Default target
all: $(TARGET)

# Build shared library for .NET P/Invoke
$(TARGET): $(SOURCE)
	@echo "🚀 Compiling TARS CUDA Vector Store..."
	@echo "📁 Source: $(SOURCE)"
	@echo "🎯 Target: $(TARGET)"
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE) $(CUDA_LIBS)
	@echo "✅ CUDA library compiled successfully!"
	@echo "📦 Output: $(TARGET)"

# Build test executable
test: $(SOURCE)
	@echo "🧪 Building CUDA test executable..."
	$(NVCC) -O3 -arch=sm_75 -o $(TEST_TARGET) $(SOURCE) $(CUDA_LIBS) -DTEST_MAIN
	@echo "✅ Test executable built: $(TEST_TARGET)"

# Run test
run-test: test
	@echo "🚀 Running CUDA Vector Store Test..."
	./$(TEST_TARGET)

# Check CUDA installation
check-cuda:
	@echo "🔍 Checking CUDA installation..."
	@nvcc --version || echo "❌ CUDA not found - install CUDA toolkit"
	@nvidia-smi || echo "❌ NVIDIA driver not found"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(TARGET) $(TEST_TARGET)
	@echo "✅ Clean complete"

# Install to system (optional)
install: $(TARGET)
	@echo "📦 Installing TARS CUDA library..."
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "✅ Library installed to /usr/local/lib/"

# Help
help:
	@echo "🚀 TARS CUDA Vector Store Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Build shared library (default)"
	@echo "  test       - Build test executable"
	@echo "  run-test   - Build and run test"
	@echo "  check-cuda - Check CUDA installation"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install library to system"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit (nvcc)"
	@echo "  - NVIDIA GPU with compute capability 7.5+"
	@echo "  - WSL with CUDA support"

.PHONY: all test run-test check-cuda clean install help
