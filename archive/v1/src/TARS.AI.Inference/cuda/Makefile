# TARS AI Inference Engine - CUDA Kernels Makefile
# Compile in WSL with real CUDA toolkit

NVCC = nvcc
CXX = g++
CUDA_PATH = /usr/local/cuda
CUDA_ARCH = -arch=sm_75  # Adjust for your GPU architecture

# Compiler flags
NVCC_FLAGS = -O3 -std=c++17 -Xcompiler -fPIC
CXX_FLAGS = -O3 -std=c++17 -fPIC -Wall -Wextra

# Include paths
INCLUDES = -I$(CUDA_PATH)/include -I./include

# Library paths
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcurand

# Source files
CUDA_SOURCES = kernels/matrix_ops.cu kernels/element_ops.cu kernels/activation_ops.cu
CPP_SOURCES = src/cuda_wrapper.cpp src/memory_manager.cpp
HEADERS = include/tars_cuda.h include/tensor_ops.h

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
ALL_OBJECTS = $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Target library
TARGET = libtars_cuda.so

# Default target
all: $(TARGET)

# Build shared library
$(TARGET): $(ALL_OBJECTS)
	@echo "🔗 Linking TARS CUDA library..."
	$(NVCC) -shared $(CUDA_ARCH) -o $@ $^ $(LIBS)
	@echo "✅ TARS CUDA library built: $(TARGET)"

# Compile CUDA source files
%.o: %.cu
	@echo "🚀 Compiling CUDA kernel: $<"
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	@echo "🔧 Compiling C++ wrapper: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(ALL_OBJECTS) $(TARGET)
	@echo "✅ Clean complete"

# Install to system (optional)
install: $(TARGET)
	@echo "📦 Installing TARS CUDA library..."
	sudo cp $(TARGET) /usr/local/lib/
	sudo ldconfig
	@echo "✅ Installation complete"

# Test CUDA setup
test-cuda:
	@echo "🧪 Testing CUDA setup..."
	nvcc --version
	nvidia-smi
	@echo "✅ CUDA test complete"

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG
debug: $(TARGET)

# Profile build
profile: NVCC_FLAGS += -lineinfo
profile: $(TARGET)

# Help
help:
	@echo "TARS AI Inference Engine - CUDA Build System"
	@echo "============================================="
	@echo "Targets:"
	@echo "  all       - Build the CUDA library (default)"
	@echo "  clean     - Clean build artifacts"
	@echo "  install   - Install library to system"
	@echo "  test-cuda - Test CUDA installation"
	@echo "  debug     - Build with debug symbols"
	@echo "  profile   - Build with profiling info"
	@echo "  help      - Show this help"

.PHONY: all clean install test-cuda debug profile help
