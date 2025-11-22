namespace TarsEngine

open System
open System.Runtime.InteropServices
open Microsoft.Extensions.Logging

/// CUDA Interop - P/Invoke bindings to TARS CUDA kernels
module CudaInterop =
    
    // ============================================================================
    // CUDA ERROR CODES
    // ============================================================================
    
    [<Struct>]
    type TarsCudaError =
        | Success = 0
        | InvalidDevice = 1
        | OutOfMemory = 2
        | InvalidValue = 3
        | KernelLaunch = 4
        | CublasError = 5
    
    // ============================================================================
    // CUDA DATA STRUCTURES
    // ============================================================================
    
    [<Struct; StructLayout(LayoutKind.Sequential)>]
    type TarsTensor =
        val mutable Data: nativeint
        val mutable Shape: nativeint
        val mutable Stride: nativeint
        val mutable NDim: int
        val mutable DType: int  // 0=float32, 1=float16, 2=bfloat16
        val mutable DeviceId: int
        val mutable SizeBytes: unativeint
    
    [<Struct; StructLayout(LayoutKind.Sequential)>]
    type TarsModelConfig =
        val mutable BatchSize: int
        val mutable SeqLen: int
        val mutable HiddenSize: int
        val mutable NumHeads: int
        val mutable NumLayers: int
        val mutable VocabSize: int
        val mutable DropoutRate: float32
        val mutable UseFlashAttention: int
    
    [<Struct; StructLayout(LayoutKind.Sequential)>]
    type TarsPerformanceMetrics =
        val mutable InferenceTimeMs: float32
        val mutable MemoryUsageMb: float32
        val mutable GpuUtilization: float32
        val mutable TensorCoreUtilization: float32
        val mutable TokensPerSecond: int
    
    // ============================================================================
    // P/INVOKE DECLARATIONS - CROSS-PLATFORM
    // ============================================================================

    // Platform-specific library loading
    module private PlatformLibrary =
        open System.Runtime.InteropServices

        let isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        let isLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
        let isMacOS = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)

        let libraryName =
            if isWindows then "TarsCudaKernels.dll"
            elif isLinux then "libTarsCudaKernels.so"
            elif isMacOS then "libTarsCudaKernels.dylib"
            else "libTarsCudaKernels.so" // Default to Linux

    // Windows P/Invoke declarations
    module private WindowsInterop =
        [<DllImport("TarsCudaKernels.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_cuda_init_win(int deviceId)

        [<DllImport("TarsCudaKernels.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_cuda_cleanup_win()

        [<DllImport("TarsCudaKernels.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_cuda_get_device_info_win(
            int deviceId,
            [<MarshalAs(UnmanagedType.LPStr)>] System.Text.StringBuilder name,
            unativeint nameLen,
            unativeint& totalMemory,
            int& computeCapability)

        [<DllImport("TarsCudaKernels.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_gemm_tensor_core_win(
            nativeint A, nativeint B, nativeint C,
            int M, int N, int K,
            float32 alpha, float32 beta, nativeint stream)

        [<DllImport("TarsCudaKernels.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_gelu_forward_win(
            nativeint input, nativeint output, int size, nativeint stream)

        [<DllImport("TarsCudaKernels.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern int tars_cuda_device_count_win()

    // Linux P/Invoke declarations
    module private LinuxInterop =
        [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_cuda_init_linux(int deviceId)

        [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_cuda_cleanup_linux()

        [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_cuda_get_device_info_linux(
            int deviceId,
            [<MarshalAs(UnmanagedType.LPStr)>] System.Text.StringBuilder name,
            unativeint nameLen,
            unativeint& totalMemory,
            int& computeCapability)

        [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_gemm_tensor_core_linux(
            nativeint A, nativeint B, nativeint C,
            int M, int N, int K,
            float32 alpha, float32 beta, nativeint stream)

        [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
        extern TarsCudaError tars_gelu_forward_linux(
            nativeint input, nativeint output, int size, nativeint stream)

        [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int tars_cuda_device_count_linux()

    // Cross-platform wrapper functions
    let tars_cuda_init(deviceId: int) =
        if PlatformLibrary.isWindows then
            WindowsInterop.tars_cuda_init_win(deviceId)
        else
            LinuxInterop.tars_cuda_init_linux(deviceId)

    let tars_cuda_cleanup() =
        if PlatformLibrary.isWindows then
            WindowsInterop.tars_cuda_cleanup_win()
        else
            LinuxInterop.tars_cuda_cleanup_linux()

    let tars_cuda_get_device_info(deviceId, name, nameLen, totalMemory, computeCapability) =
        if PlatformLibrary.isWindows then
            WindowsInterop.tars_cuda_get_device_info_win(deviceId, name, nameLen, &totalMemory, &computeCapability)
        else
            LinuxInterop.tars_cuda_get_device_info_linux(deviceId, name, nameLen, &totalMemory, &computeCapability)

    let tars_gemm_tensor_core(A, B, C, M, N, K, alpha, beta, stream) =
        if PlatformLibrary.isWindows then
            WindowsInterop.tars_gemm_tensor_core_win(A, B, C, M, N, K, alpha, beta, stream)
        else
            LinuxInterop.tars_gemm_tensor_core_linux(A, B, C, M, N, K, alpha, beta, stream)

    let tars_gelu_forward(input, output, size, stream) =
        if PlatformLibrary.isWindows then
            WindowsInterop.tars_gelu_forward_win(input, output, size, stream)
        else
            LinuxInterop.tars_gelu_forward_linux(input, output, size, stream)

    let tars_cuda_device_count() =
        if PlatformLibrary.isWindows then
            WindowsInterop.tars_cuda_device_count_win()
        else
            LinuxInterop.tars_cuda_device_count_linux()

    // Legacy single declarations for compatibility
    [<System.Obsolete("Use cross-platform wrapper functions instead")>]
    extern TarsCudaError tars_cuda_init(int deviceId)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_cleanup()

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_get_device_info(
        int deviceId, 
        [<MarshalAs(UnmanagedType.LPStr)>] System.Text.StringBuilder name, 
        unativeint nameLen,
        unativeint& totalMemory,
        int& computeCapability)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_malloc(nativeint& ptr, unativeint size)

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_free(nativeint ptr)

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_memcpy_h2d(nativeint dst, nativeint src, unativeint size)

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_memcpy_d2h(nativeint dst, nativeint src, unativeint size)

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_tensor_create(
        TarsTensor& tensor, 
        int[] shape, 
        int ndim, 
        int dtype, 
        int deviceId)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_tensor_destroy(TarsTensor& tensor)

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_gemm_tensor_core(
        nativeint A, 
        nativeint B, 
        nativeint C,
        int M, 
        int N, 
        int K,
        float32 alpha, 
        float32 beta,
        nativeint stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_gelu_forward(
        nativeint input,
        nativeint output,
        int size,
        nativeint stream)

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int tars_cuda_device_count()

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_set_device(int deviceId)

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_synchronize_device()

    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint tars_cuda_get_error_string(TarsCudaError error)
    
    // ============================================================================
    // F# WRAPPER FUNCTIONS
    // ============================================================================
    
    /// CUDA device information
    type CudaDeviceInfo = {
        DeviceId: int
        Name: string
        TotalMemory: uint64
        ComputeCapability: int
        IsAvailable: bool
    }
    
    /// CUDA tensor wrapper
    type CudaTensorWrapper = {
        Tensor: TarsTensor
        Shape: int array
        DataType: string
    }
    
    /// CUDA context manager
    type CudaContext(deviceId: int, logger: ILogger<CudaContext>) =
        let mutable isInitialized = false
        let mutable deviceInfo: CudaDeviceInfo option = None
        
        /// Initialize CUDA context
        member _.Initialize() = async {
            logger.LogInformation($"🔧 Initializing CUDA context on device {deviceId}...")
            
            let result = tars_cuda_init(deviceId)
            if result = TarsCudaError.Success then
                // Get device information
                let name = System.Text.StringBuilder(256)
                let mutable totalMemory = 0UL
                let mutable computeCapability = 0
                
                let infoResult = tars_cuda_get_device_info(deviceId, name, 256UL, &totalMemory, &computeCapability)
                if infoResult = TarsCudaError.Success then
                    deviceInfo <- Some {
                        DeviceId = deviceId
                        Name = name.ToString()
                        TotalMemory = totalMemory
                        ComputeCapability = computeCapability
                        IsAvailable = true
                    }
                    
                    isInitialized <- true
                    logger.LogInformation($"✅ CUDA initialized: {name} ({totalMemory / (1024UL * 1024UL * 1024UL)}GB, CC {computeCapability / 10}.{computeCapability % 10})")
                    return true
                else
                    logger.LogError($"❌ Failed to get device info: {infoResult}")
                    return false
            else
                logger.LogError($"❌ Failed to initialize CUDA: {result}")
                return false
        }
        
        /// Get device information
        member _.GetDeviceInfo() = deviceInfo
        
        /// Check if CUDA is initialized
        member _.IsInitialized = isInitialized
        
        /// Create tensor on GPU
        member _.CreateTensor(shape: int array, dataType: string) = async {
            if not isInitialized then
                failwith "CUDA context not initialized"
            
            let dtype = 
                match dataType.ToLower() with
                | "float32" -> 0
                | "float16" -> 1
                | "bfloat16" -> 2
                | _ -> failwith $"Unsupported data type: {dataType}"
            
            let mutable tensor = TarsTensor()
            let result = tars_tensor_create(&tensor, shape, shape.Length, dtype, deviceId)
            
            if result = TarsCudaError.Success then
                logger.LogInformation($"✅ Created tensor: {shape} ({dataType})")
                return Some {
                    Tensor = tensor
                    Shape = shape
                    DataType = dataType
                }
            else
                logger.LogError($"❌ Failed to create tensor: {result}")
                return None
        }
        
        /// Destroy tensor
        member _.DestroyTensor(tensorWrapper: CudaTensorWrapper) = async {
            let mutable tensor = tensorWrapper.Tensor
            let result = tars_tensor_destroy(&tensor)
            
            if result = TarsCudaError.Success then
                logger.LogInformation("✅ Tensor destroyed")
                return true
            else
                logger.LogError($"❌ Failed to destroy tensor: {result}")
                return false
        }
        
        /// Run matrix multiplication with Tensor Cores
        member _.RunGemmTensorCore(A: nativeint, B: nativeint, C: nativeint, M: int, N: int, K: int, alpha: float32, beta: float32) = async {
            logger.LogInformation($"⚡ Running GEMM with Tensor Cores: {M}x{N}x{K}")
            
            let result = tars_gemm_tensor_core(A, B, C, M, N, K, alpha, beta, nativeint 0)
            
            if result = TarsCudaError.Success then
                // Synchronize to measure actual execution time
                let syncResult = tars_synchronize_device()
                if syncResult = TarsCudaError.Success then
                    logger.LogInformation("✅ GEMM completed successfully")
                    return true
                else
                    logger.LogError($"❌ Failed to synchronize: {syncResult}")
                    return false
            else
                logger.LogError($"❌ GEMM failed: {result}")
                return false
        }
        
        /// Run GELU activation
        member _.RunGeluActivation(input: nativeint, output: nativeint, size: int) = async {
            logger.LogInformation($"⚡ Running GELU activation: {size} elements")
            
            let result = tars_gelu_forward(input, output, size, nativeint 0)
            
            if result = TarsCudaError.Success then
                let syncResult = tars_synchronize_device()
                if syncResult = TarsCudaError.Success then
                    logger.LogInformation("✅ GELU activation completed")
                    return true
                else
                    logger.LogError($"❌ Failed to synchronize: {syncResult}")
                    return false
            else
                logger.LogError($"❌ GELU activation failed: {result}")
                return false
        }
        
        /// Cleanup CUDA context
        member _.Cleanup() = async {
            if isInitialized then
                logger.LogInformation("🧹 Cleaning up CUDA context...")
                let result = tars_cuda_cleanup()
                
                if result = TarsCudaError.Success then
                    isInitialized <- false
                    logger.LogInformation("✅ CUDA cleanup complete")
                    return true
                else
                    logger.LogError($"❌ CUDA cleanup failed: {result}")
                    return false
            else
                return true
        }
        
        interface IDisposable with
            member this.Dispose() =
                this.Cleanup() |> Async.RunSynchronously |> ignore
    
    /// Get available CUDA devices
    let getAvailableCudaDevices() = async {
        let deviceCount = tars_cuda_device_count()
        
        let devices = Array.zeroCreate deviceCount
        for i in 0 .. deviceCount - 1 do
            let name = System.Text.StringBuilder(256)
            let mutable totalMemory = 0UL
            let mutable computeCapability = 0
            
            let result = tars_cuda_get_device_info(i, name, 256UL, &totalMemory, &computeCapability)
            devices.[i] <- {
                DeviceId = i
                Name = if result = TarsCudaError.Success then name.ToString() else "Unknown"
                TotalMemory = totalMemory
                ComputeCapability = computeCapability
                IsAvailable = result = TarsCudaError.Success
            }
        
        return devices
    }
