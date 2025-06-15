namespace TarsEngine.FSharp.Core.GPU

open System
open System.Runtime.InteropServices

/// P/Invoke bindings for TARS CUDA reasoning kernels
module CudaReasoningBindings =
    
    // ============================================================================
    // CUDA ERROR HANDLING
    // ============================================================================
    
    [<Struct>]
    type TarsCudaError =
        | Success = 0
        | InvalidDevice = 1
        | MemoryAllocation = 2
        | KernelLaunch = 3
        | InvalidParameter = 4
        | CublasError = 5
        | CudnnError = 6
        | UnsupportedOperation = 7
    
    // ============================================================================
    // CUDA STREAM HANDLE
    // ============================================================================
    
    type CudaStream = nativeint
    
    // ============================================================================
    // SEDENION OPERATIONS
    // ============================================================================
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_sedenion_distance(
        nativeint vectors1,      // const float* vectors1
        nativeint vectors2,      // const float* vectors2  
        nativeint distances,     // float* distances
        int num_vectors,         // int num_vectors
        int dimensions,          // int dimensions
        CudaStream stream)       // cudaStream_t stream
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_sedenion_multiply(
        nativeint a,             // const float* a
        nativeint b,             // const float* b
        nativeint result,        // float* result
        int num_operations,      // int num_operations
        int dimensions,          // int dimensions
        CudaStream stream)       // cudaStream_t stream
    
    // ============================================================================
    // CROSS ENTROPY OPERATIONS
    // ============================================================================
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cross_entropy_loss(
        nativeint predictions,   // const float* predictions
        nativeint targets,       // const float* targets
        nativeint losses,        // float* losses
        int batch_size,          // int batch_size
        int num_classes,         // int num_classes
        CudaStream stream)       // cudaStream_t stream
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_softmax_activation(
        nativeint input,         // const float* input
        nativeint output,        // float* output
        int batch_size,          // int batch_size
        int num_classes,         // int num_classes
        CudaStream stream)       // cudaStream_t stream
    
    // ============================================================================
    // MARKOV CHAIN OPERATIONS
    // ============================================================================
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_markov_transition(
        nativeint states,        // const float* states
        nativeint transition_matrix, // const float* transition_matrix
        nativeint next_states,   // float* next_states
        int num_chains,          // int num_chains
        int num_states,          // int num_states
        CudaStream stream)       // cudaStream_t stream
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_markov_steady_state(
        nativeint transition_matrix, // const float* transition_matrix
        nativeint steady_state,  // float* steady_state
        int num_states,          // int num_states
        int max_iterations,      // int max_iterations
        float tolerance,         // float tolerance
        CudaStream stream)       // cudaStream_t stream
    
    // ============================================================================
    // NEURAL NETWORK OPERATIONS
    // ============================================================================
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_neural_forward_pass(
        nativeint input,         // const float* input
        nativeint weights,       // const float* weights
        nativeint bias,          // const float* bias
        nativeint output,        // float* output
        int batch_size,          // int batch_size
        int input_size,          // int input_size
        int output_size,         // int output_size
        CudaStream stream)       // cudaStream_t stream
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_relu_activation(
        nativeint input,         // const float* input
        nativeint output,        // float* output
        int size,                // int size
        CudaStream stream)       // cudaStream_t stream
    
    // ============================================================================
    // PERFORMANCE MEASUREMENT
    // ============================================================================
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_flops_benchmark(
        nativeint input,         // const float* input
        nativeint output,        // float* output
        int size,                // int size
        int operations_per_element, // int operations_per_element
        CudaStream stream)       // cudaStream_t stream
    
    [<DllImport("TarsReasoningKernels", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_memory_bandwidth_benchmark(
        nativeint input,         // const float* input
        nativeint output,        // float* output
        int size,                // int size
        CudaStream stream)       // cudaStream_t stream
    
    // ============================================================================
    // CUDA MEMORY MANAGEMENT
    // ============================================================================
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaMalloc(nativeint* devPtr, uint64 size)
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaFree(nativeint devPtr)
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaMemcpy(nativeint dst, nativeint src, uint64 count, int kind)
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaMemcpyAsync(nativeint dst, nativeint src, uint64 count, int kind, CudaStream stream)
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaStreamCreate(CudaStream* stream)
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaStreamDestroy(CudaStream stream)
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaStreamSynchronize(CudaStream stream)
    
    [<DllImport("cudart64_12", CallingConvention = CallingConvention.Cdecl)>]
    extern int cudaDeviceSynchronize()
    
    // ============================================================================
    // MEMORY COPY KINDS
    // ============================================================================
    
    [<Literal>]
    let cudaMemcpyHostToDevice = 1
    
    [<Literal>]
    let cudaMemcpyDeviceToHost = 2
    
    [<Literal>]
    let cudaMemcpyDeviceToDevice = 3
    
    // ============================================================================
    // HIGH-LEVEL F# WRAPPER FUNCTIONS
    // ============================================================================
    
    /// Allocate GPU memory and return pointer
    let allocateGpuMemory (sizeInBytes: uint64) : Result<nativeint, string> =
        let mutable devPtr = nativeint 0
        let result = cudaMalloc(&devPtr, sizeInBytes)
        if result = 0 then
            Ok devPtr
        else
            Error $"CUDA memory allocation failed with error code {result}"
    
    /// Free GPU memory
    let freeGpuMemory (devPtr: nativeint) : Result<unit, string> =
        let result = cudaFree(devPtr)
        if result = 0 then
            Ok ()
        else
            Error $"CUDA memory free failed with error code {result}"
    
    /// Copy data from host to device
    let copyHostToDevice (hostPtr: nativeint) (devicePtr: nativeint) (sizeInBytes: uint64) : Result<unit, string> =
        let result = cudaMemcpy(devicePtr, hostPtr, sizeInBytes, cudaMemcpyHostToDevice)
        if result = 0 then
            Ok ()
        else
            Error $"CUDA host to device copy failed with error code {result}"
    
    /// Copy data from device to host
    let copyDeviceToHost (devicePtr: nativeint) (hostPtr: nativeint) (sizeInBytes: uint64) : Result<unit, string> =
        let result = cudaMemcpy(hostPtr, devicePtr, sizeInBytes, cudaMemcpyDeviceToHost)
        if result = 0 then
            Ok ()
        else
            Error $"CUDA device to host copy failed with error code {result}"
    
    /// Create CUDA stream
    let createStream () : Result<CudaStream, string> =
        let mutable stream = nativeint 0
        let result = cudaStreamCreate(&stream)
        if result = 0 then
            Ok stream
        else
            Error $"CUDA stream creation failed with error code {result}"
    
    /// Destroy CUDA stream
    let destroyStream (stream: CudaStream) : Result<unit, string> =
        let result = cudaStreamDestroy(stream)
        if result = 0 then
            Ok ()
        else
            Error $"CUDA stream destruction failed with error code {result}"
    
    /// Synchronize CUDA stream
    let synchronizeStream (stream: CudaStream) : Result<unit, string> =
        let result = cudaStreamSynchronize(stream)
        if result = 0 then
            Ok ()
        else
            Error $"CUDA stream synchronization failed with error code {result}"
    
    /// Synchronize device
    let synchronizeDevice () : Result<unit, string> =
        let result = cudaDeviceSynchronize()
        if result = 0 then
            Ok ()
        else
            Error $"CUDA device synchronization failed with error code {result}"
