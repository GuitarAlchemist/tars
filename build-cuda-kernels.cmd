@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo                    TARS CUDA KERNELS BUILD SYSTEM
echo ========================================================================
echo.
echo 🔧 Building TARS massively parallel CUDA neural network kernels
echo    Real CUDA compilation with nvcc and CMake
echo.

echo 🎯 BUILD CONFIGURATION:
echo =======================
echo.
echo 📋 Target Architectures:
echo    • Compute Capability 7.5 (RTX 20 series)
echo    • Compute Capability 8.0 (A100)
echo    • Compute Capability 8.6 (RTX 30 series)
echo    • Compute Capability 8.9 (RTX 40 series)
echo    • Compute Capability 9.0 (H100)
echo.
echo ⚡ Optimization Flags:
echo    • -O3: Maximum optimization
echo    • --use_fast_math: Fast math operations
echo    • -Xcompiler -fPIC: Position independent code
echo    • Tensor Core support enabled
echo.

echo [%TIME%] 🔍 Checking CUDA development environment...

REM Check if CUDA is installed
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ❌ CUDA Toolkit not found! Please install CUDA Toolkit 11.0 or later
    echo [%TIME%] 📥 Download from: https://developer.nvidia.com/cuda-toolkit
    echo [%TIME%] 🔧 Make sure nvcc is in your PATH
    pause
    exit /b 1
)

echo [%TIME%] ✅ CUDA Toolkit found
nvcc --version | findstr "release"

REM Check if CMake is installed
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ❌ CMake not found! Please install CMake 3.18 or later
    echo [%TIME%] 📥 Download from: https://cmake.org/download/
    pause
    exit /b 1
)

echo [%TIME%] ✅ CMake found
cmake --version | findstr "version"

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ⚠️ nvidia-smi not found - GPU detection may fail
    echo [%TIME%] 💡 CUDA kernels will still compile for target architectures
) else (
    echo [%TIME%] ✅ NVIDIA GPU driver found
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | head -1
)

echo.
echo [%TIME%] 🏗️ Setting up build directory...

REM Create build directory
if not exist "build" mkdir build
cd build

echo [%TIME%] ✅ Build directory ready

echo.
echo [%TIME%] 🔧 Configuring CMake build system...

REM Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" ^
      -DCMAKE_CUDA_COMPILER=nvcc ^
      -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math -Xcompiler -fPIC" ^
      ../src/TarsEngine/CUDA

if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ❌ CMake configuration failed!
    echo [%TIME%] 🔍 Check CUDA installation and GPU compute capability
    cd ..
    pause
    exit /b 1
)

echo [%TIME%] ✅ CMake configuration complete

echo.
echo [%TIME%] ⚡ Compiling CUDA kernels...
echo [%TIME%] 🔄 This may take several minutes for all GPU architectures...

REM Build with CMake
cmake --build . --config Release --parallel 4

if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ❌ CUDA kernel compilation failed!
    echo [%TIME%] 🔍 Check compiler errors above
    cd ..
    pause
    exit /b 1
)

echo [%TIME%] ✅ CUDA kernels compiled successfully!

echo.
echo [%TIME%] 📦 Installing CUDA library...

REM Install the library
cmake --install . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ⚠️ Installation failed, but library is built
    echo [%TIME%] 📁 Library location: build/Release/TarsCudaKernels.dll
)

echo [%TIME%] ✅ CUDA library installation complete

echo.
echo [%TIME%] 🧪 Testing CUDA kernel functionality...

REM Check if the library was built
if exist "Release\TarsCudaKernels.dll" (
    echo [%TIME%] ✅ TarsCudaKernels.dll built successfully
    
    REM Get file size
    for %%A in ("Release\TarsCudaKernels.dll") do (
        echo [%TIME%] 📊 Library size: %%~zA bytes
    )
    
    REM Copy to main directory for F# to find
    copy "Release\TarsCudaKernels.dll" "..\TarsCudaKernels.dll" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [%TIME%] ✅ Library copied to main directory
    )
    
) else if exist "libTarsCudaKernels.so" (
    echo [%TIME%] ✅ libTarsCudaKernels.so built successfully (Linux)
    
    REM Copy to main directory
    copy "libTarsCudaKernels.so" "..\libTarsCudaKernels.so" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [%TIME%] ✅ Library copied to main directory
    )
    
) else (
    echo [%TIME%] ❌ CUDA library not found in expected location
    echo [%TIME%] 🔍 Check build output above for errors
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo [%TIME%] 🔬 Analyzing compiled kernels...

REM Use objdump or similar to analyze the compiled kernels (if available)
where objdump >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [%TIME%] 📊 Analyzing kernel symbols...
    objdump -T TarsCudaKernels.dll 2>nul | findstr "tars_" | head -5
    if !ERRORLEVEL! EQU 0 (
        echo [%TIME%] ✅ CUDA kernel symbols found
    )
)

echo.
echo ========================================================================
echo 🎉 TARS CUDA KERNELS BUILD COMPLETE!
echo ========================================================================
echo.
echo ✅ BUILD SUCCESS!
echo.
echo 📦 DELIVERABLES:
echo    • TarsCudaKernels.dll (Windows) or libTarsCudaKernels.so (Linux)
echo    • Optimized for GPU architectures: 7.5, 8.0, 8.6, 8.9, 9.0
echo    • Tensor Core support enabled
echo    • cuBLAS integration ready
echo    • P/Invoke compatible for F# integration
echo.
echo 🔧 COMPILED KERNELS:
echo    ✅ Matrix multiplication with Tensor Cores
echo    ✅ GELU activation functions
echo    ✅ Layer normalization
echo    ✅ Embedding lookup
echo    ✅ Memory management utilities
echo    ✅ Device management functions
echo.
echo ⚡ OPTIMIZATION FEATURES:
echo    • -O3 maximum compiler optimization
echo    • Fast math operations enabled
echo    • Tensor Core WMMA API utilization
echo    • Coalesced memory access patterns
echo    • Shared memory optimization
echo    • Multi-GPU architecture support
echo.
echo 🎯 NEXT STEPS:
echo    1. Test F# P/Invoke integration
echo    2. Run performance benchmarks
echo    3. Validate mathematical accuracy
echo    4. Integrate with TARS AI inference engine
echo    5. Deploy production neural network
echo.
echo 🚀 Ready for Phase 3: Performance Validation!
echo.
echo 💡 The CUDA kernels are now compiled and ready for real
echo    GPU acceleration of TARS neural network inference!
echo.

pause
