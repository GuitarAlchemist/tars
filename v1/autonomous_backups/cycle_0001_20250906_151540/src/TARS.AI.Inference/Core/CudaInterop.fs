namespace TARS.AI.Inference.Core

open System
open System.Runtime.InteropServices

/// P/Invoke bindings for TARS CUDA library
/// Real CUDA implementation compiled in WSL
module CudaInterop =

    // ============================================================================
    // Native Types and Structures
    // ============================================================================

    [<StructLayout(LayoutKind.Sequential)>]
    type TarsTensor = {
        mutable Data: IntPtr
        mutable Shape: IntPtr
        mutable NDim: int
        mutable TotalSize: int
        mutable IsOnGPU: bool
    }

    [<StructLayout(LayoutKind.Sequential)>]
    type TarsCudaContext = {
        mutable Stream: IntPtr
        mutable CublasHandle: IntPtr
        mutable CurandGen: IntPtr
        mutable DeviceId: int
        mutable Initialized: bool
    }

    type TarsError =
        | Success = 0
        | CudaRuntime = 1
        | Cublas = 2
        | InvalidParams = 3
        | OutOfMemory = 4
        | NotInitialized = 5

    // ============================================================================
    // Native Library Path
    // ============================================================================

    [<Literal>]
    let private libraryName = "libtars_cuda"

    // ============================================================================
    // Context Management
    // ============================================================================

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_init(TarsCudaContext& ctx, int deviceId)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_cleanup(TarsCudaContext& ctx)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_get_device_info(int deviceId, IntPtr name, IntPtr& memoryMb)

    // ============================================================================
    // Memory Management
    // ============================================================================

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_alloc_tensor(TarsTensor& tensor, int size)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_free_tensor(TarsTensor& tensor)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_copy_to_gpu(TarsTensor& tensor, float[] hostData, int size)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_copy_to_cpu(TarsTensor& tensor, float[] hostData, int size)

    // ============================================================================
    // Matrix Operations
    // ============================================================================

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_matrix_multiply(TarsCudaContext& ctx, 
                                              TarsTensor& A, 
                                              TarsTensor& B, 
                                              TarsTensor& C,
                                              int M, int N, int K)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_batch_matrix_multiply(TarsCudaContext& ctx,
                                                    TarsTensor& A,
                                                    TarsTensor& B,
                                                    TarsTensor& C,
                                                    int batchSize, int M, int N, int K)

    // ============================================================================
    // Element-wise Operations
    // ============================================================================

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_element_add(TarsCudaContext& ctx,
                                          TarsTensor& A,
                                          TarsTensor& B,
                                          TarsTensor& C,
                                          int size)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_element_multiply(TarsCudaContext& ctx,
                                              TarsTensor& A,
                                              TarsTensor& B,
                                              TarsTensor& C,
                                              int size)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_scalar_multiply(TarsCudaContext& ctx,
                                             TarsTensor& A,
                                             TarsTensor& B,
                                             float alpha,
                                             int size)

    // ============================================================================
    // Activation Functions
    // ============================================================================

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_relu(TarsCudaContext& ctx,
                                   TarsTensor& A,
                                   TarsTensor& B,
                                   int size)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_gelu(TarsCudaContext& ctx,
                                   TarsTensor& A,
                                   TarsTensor& B,
                                   int size)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_softmax(TarsCudaContext& ctx,
                                      TarsTensor& A,
                                      TarsTensor& B,
                                      int rows, int cols)

    // ============================================================================
    // Transformer Operations
    // ============================================================================

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_multi_head_attention(TarsCudaContext& ctx,
                                                   TarsTensor& Q,
                                                   TarsTensor& K,
                                                   TarsTensor& V,
                                                   TarsTensor& output,
                                                   int batchSize, int seqLen,
                                                   int numHeads, int headDim)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_layer_norm(TarsCudaContext& ctx,
                                         TarsTensor& input,
                                         TarsTensor& output,
                                         TarsTensor& gamma,
                                         TarsTensor& beta,
                                         int batchSize, int hiddenSize,
                                         float eps)

    // ============================================================================
    // Utility Functions
    // ============================================================================

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern IntPtr tars_cuda_get_error_string(TarsError error)

    [<DllImport(libraryName, CallingConvention = CallingConvention.Cdecl)>]
    extern TarsError tars_cuda_synchronize(TarsCudaContext& ctx)

    // ============================================================================
    // High-Level F# Wrapper Functions
    // ============================================================================

    /// Initialize CUDA context with error handling
    let initializeCudaContext (deviceId: int) : Result<TarsCudaContext, string> =
        try
            let mutable ctx = Unchecked.defaultof<TarsCudaContext>
            let result = tars_cuda_init(&ctx, deviceId)
            
            match result with
            | TarsError.Success ->
                printfn "🚀 TARS CUDA Context Initialized"
                printfn "   Device ID: %d" deviceId
                Ok(ctx)
            | error ->
                let errorPtr = tars_cuda_get_error_string(error)
                let errorMsg = Marshal.PtrToStringAnsi(errorPtr)
                Error($"CUDA initialization failed: {errorMsg}")
        with
        | ex -> Error($"Exception during CUDA initialization: {ex.Message}")

    /// Cleanup CUDA context
    let cleanupCudaContext (ctx: TarsCudaContext) : Result<unit, string> =
        try
            let mutable mutableCtx = ctx
            let result = tars_cuda_cleanup(&mutableCtx)
            
            match result with
            | TarsError.Success ->
                printfn "✅ TARS CUDA Context Cleaned Up"
                Ok(())
            | error ->
                let errorPtr = tars_cuda_get_error_string(error)
                let errorMsg = Marshal.PtrToStringAnsi(errorPtr)
                Error($"CUDA cleanup failed: {errorMsg}")
        with
        | ex -> Error($"Exception during CUDA cleanup: {ex.Message}")

    /// Get GPU device information
    let getDeviceInfo (deviceId: int) : Result<string * int64, string> =
        try
            let nameBuffer = Marshal.AllocHGlobal(256)
            let mutable memoryMb = IntPtr.Zero
            
            let result = tars_cuda_get_device_info(deviceId, nameBuffer, &memoryMb)
            
            match result with
            | TarsError.Success ->
                let deviceName = Marshal.PtrToStringAnsi(nameBuffer)
                let memorySize = memoryMb.ToInt64()
                Marshal.FreeHGlobal(nameBuffer)
                Ok(deviceName, memorySize)
            | error ->
                Marshal.FreeHGlobal(nameBuffer)
                let errorPtr = tars_cuda_get_error_string(error)
                let errorMsg = Marshal.PtrToStringAnsi(errorPtr)
                Error($"Failed to get device info: {errorMsg}")
        with
        | ex -> Error($"Exception getting device info: {ex.Message}")

    /// Check if CUDA library is available
    let isCudaAvailable () : bool =
        try
            match getDeviceInfo(0) with
            | Ok(_) -> true
            | Error(_) -> false
        with
        | _ -> false

    /// Get error message from TarsError
    let getErrorMessage (error: TarsError) : string =
        try
            let errorPtr = tars_cuda_get_error_string(error)
            Marshal.PtrToStringAnsi(errorPtr)
        with
        | _ -> $"Unknown error: {error}"
