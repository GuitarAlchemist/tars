namespace TarsEngine.FSharp.Cli.Core

open System
open System.Runtime.InteropServices
open Microsoft.Extensions.Logging

/// CUDA computational expressions for TARS DSL integration
module CudaComputationExpression =
    
    // CUDA interop declarations
    module CudaInterop =
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int minimal_cuda_device_count()
        
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int minimal_cuda_init(int device_id)
        
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int minimal_cuda_cleanup()
        
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int cuda_simple_vector_test()
        
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int cuda_matrix_multiply_test()
        
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int cuda_neural_network_test()
        
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int cuda_transformer_attention_test()
        
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int cuda_ai_model_inference_test()
    
    /// CUDA operation result
    type CudaResult<'T> = 
        | Success of 'T
        | Error of string
        
    /// CUDA operation context
    type CudaContext = {
        DeviceId: int
        IsInitialized: bool
        Logger: ILogger option
    }
    
    /// CUDA operation types
    type CudaOperation<'T> = CudaContext -> Async<CudaResult<'T>>
    
    /// CUDA computation expression builder
    type CudaBuilder(logger: ILogger option) =
        
        /// Create default CUDA context
        let createContext deviceId = {
            DeviceId = deviceId
            IsInitialized = false
            Logger = logger
        }
        
        /// Initialize CUDA context
        let initializeCuda context =
            async {
                try
                    let result = CudaInterop.minimal_cuda_init(context.DeviceId)
                    if result = 0 then
                        context.Logger |> Option.iter (fun log -> log.LogInformation($"CUDA initialized on device {context.DeviceId}"))
                        return Success { context with IsInitialized = true }
                    else
                        let error = $"CUDA initialization failed with code: {result}"
                        context.Logger |> Option.iter (fun log -> log.LogError(error))
                        return Error error
                with
                | ex ->
                    let error = $"CUDA initialization exception: {ex.Message}"
                    context.Logger |> Option.iter (fun log -> log.LogError(ex, error))
                    return Error error
            }
        
        /// Cleanup CUDA context
        let cleanupCuda context =
            async {
                try
                    if context.IsInitialized then
                        let result = CudaInterop.minimal_cuda_cleanup()
                        context.Logger |> Option.iter (fun log -> log.LogInformation("CUDA cleanup completed"))
                        return Success ()
                    else
                        return Success ()
                with
                | ex ->
                    let error = $"CUDA cleanup exception: {ex.Message}"
                    context.Logger |> Option.iter (fun log -> log.LogError(ex, error))
                    return Error error
            }
        
        /// Execute CUDA operation with automatic context management
        let executeCudaOperation (operation: CudaContext -> Async<CudaResult<'T>>) deviceId =
            async {
                let context = createContext deviceId
                let! initResult = initializeCuda context
                
                match initResult with
                | Success initializedContext ->
                    try
                        let! operationResult = operation initializedContext
                        let! _ = cleanupCuda initializedContext
                        return operationResult
                    with
                    | ex ->
                        let! _ = cleanupCuda initializedContext
                        let error = $"CUDA operation exception: {ex.Message}"
                        context.Logger |> Option.iter (fun log -> log.LogError(ex, error))
                        return Error error
                | Error error ->
                    return Error error
            }
        
        // Computation expression methods
        member _.Return(value: 'T) : CudaOperation<'T> =
            fun context -> async { return Success value }
        
        member _.ReturnFrom(operation: CudaOperation<'T>) : CudaOperation<'T> =
            operation
        
        member _.Bind(operation: CudaOperation<'T>, continuation: 'T -> CudaOperation<'U>) : CudaOperation<'U> =
            fun context ->
                async {
                    let! result = operation context
                    match result with
                    | Success value ->
                        let nextOperation = continuation value
                        return! nextOperation context
                    | Error error ->
                        return Error error
                }
        
        member _.Zero() : CudaOperation<unit> =
            fun context -> async { return Success () }
        
        member _.Delay(f: unit -> CudaOperation<'T>) : CudaOperation<'T> =
            fun context -> f() context
        
        member _.Combine(first: CudaOperation<unit>, second: CudaOperation<'T>) : CudaOperation<'T> =
            fun context ->
                async {
                    let! firstResult = first context
                    match firstResult with
                    | Success () ->
                        return! second context
                    | Error error ->
                        return Error error
                }
        
        member _.TryWith(operation: CudaOperation<'T>, handler: exn -> CudaOperation<'T>) : CudaOperation<'T> =
            fun context ->
                async {
                    try
                        return! operation context
                    with
                    | ex ->
                        return! handler ex context
                }
        
        member _.TryFinally(operation: CudaOperation<'T>, finalizer: unit -> unit) : CudaOperation<'T> =
            fun context ->
                async {
                    try
                        return! operation context
                    finally
                        finalizer()
                }
        
        /// Execute the CUDA computation with device management
        member _.Run(operation: CudaOperation<'T>, ?deviceId: int) : Async<CudaResult<'T>> =
            let device = defaultArg deviceId 0
            executeCudaOperation operation device
    
    /// CUDA operations for TARS DSL
    module CudaOperations =
        
        /// Vector operations
        let vectorAdd () : CudaOperation<string> =
            fun context ->
                async {
                    try
                        let result = CudaInterop.cuda_simple_vector_test()
                        if result = 0 then
                            return Success "Vector addition completed successfully (1024 elements)"
                        else
                            return Error "Vector addition failed"
                    with
                    | ex ->
                        return Error $"Vector operation exception: {ex.Message}"
                }
        
        /// Matrix operations
        let matrixMultiply () : CudaOperation<string> =
            fun context ->
                async {
                    try
                        let result = CudaInterop.cuda_matrix_multiply_test()
                        if result = 0 then
                            return Success "Matrix multiplication completed successfully (64x64 matrices)"
                        else
                            return Error "Matrix multiplication failed"
                    with
                    | ex ->
                        return Error $"Matrix operation exception: {ex.Message}"
                }
        
        /// Neural network operations
        let neuralNetwork () : CudaOperation<string> =
            fun context ->
                async {
                    try
                        let result = CudaInterop.cuda_neural_network_test()
                        if result = 0 then
                            return Success "Neural network operations completed (ReLU, Sigmoid activations)"
                        else
                            return Error "Neural network operations failed"
                    with
                    | ex ->
                        return Error $"Neural network exception: {ex.Message}"
                }
        
        /// Transformer attention
        let transformerAttention () : CudaOperation<string> =
            fun context ->
                async {
                    try
                        let result = CudaInterop.cuda_transformer_attention_test()
                        if result = 0 then
                            return Success "Transformer attention computed successfully"
                        else
                            return Error "Transformer attention failed"
                    with
                    | ex ->
                        return Error $"Transformer attention exception: {ex.Message}"
                }
        
        /// AI model inference
        let aiModelInference () : CudaOperation<string> =
            fun context ->
                async {
                    try
                        let result = CudaInterop.cuda_ai_model_inference_test()
                        if result = 0 then
                            return Success "AI model inference pipeline completed successfully"
                        else
                            return Error "AI model inference failed"
                    with
                    | ex ->
                        return Error $"AI model inference exception: {ex.Message}"
                }
        
        /// Get device information
        let getDeviceInfo () : CudaOperation<string> =
            fun context ->
                async {
                    try
                        let deviceCount = CudaInterop.minimal_cuda_device_count()
                        return Success $"CUDA devices available: {deviceCount}"
                    with
                    | ex ->
                        return Error $"Device info exception: {ex.Message}"
                }
    
    /// Create CUDA computation expression instance
    let cuda (logger: ILogger option) = CudaBuilder(logger)
    
    /// Create CUDA computation expression with default logger
    let cudaDefault = CudaBuilder(None)
