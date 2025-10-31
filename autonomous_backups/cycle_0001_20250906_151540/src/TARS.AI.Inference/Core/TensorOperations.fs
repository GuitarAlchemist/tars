namespace TARS.AI.Inference.Core

open System
open System.Numerics.Tensors
open ILGPU
open ILGPU.Runtime

/// High-performance tensor operations with CUDA acceleration
module TensorOperations =

    /// TARS custom tensor type with CUDA support
    type TarsTensor<'T when 'T : struct> = {
        Data: 'T[]
        Shape: int[]
        Device: MemoryBuffer1D<'T, Stride1D.Dense> option
        IsOnGPU: bool
    }

    /// CUDA context for GPU operations
    type CudaContext = {
        Context: Context
        Accelerator: Accelerator
        Stream: AcceleratorStream
    }

    /// Initialize CUDA context for tensor operations
    let initializeCuda () =
        try
            let context = Context.CreateDefault()
            let accelerator = context.GetPreferredDevice(preferGPU = true).CreateAccelerator(context)
            let stream = accelerator.CreateStream()
            
            printfn "🚀 TARS CUDA Initialized: %s" accelerator.Name
            printfn "   Memory: %d MB" (accelerator.MemorySize / (1024L * 1024L))
            
            Some({
                Context = context
                Accelerator = accelerator
                Stream = stream
            })
        with
        | ex ->
            printfn "⚠️ CUDA initialization failed: %s" ex.Message
            printfn "   Falling back to CPU operations"
            None

    /// Create a new tensor with specified shape
    let createTensor<'T when 'T : struct> (shape: int[]) (data: 'T[]) (cudaContext: CudaContext option) : TarsTensor<'T> =
        let totalSize = Array.fold (*) 1 shape
        if data.Length <> totalSize then
            failwith $"Data length {data.Length} doesn't match tensor size {totalSize}"
        
        match cudaContext with
        | Some ctx ->
            try
                let buffer = ctx.Accelerator.Allocate1D<'T>(data.Length)
                buffer.CopyFromCPU(data)
                {
                    Data = data
                    Shape = shape
                    Device = Some(buffer)
                    IsOnGPU = true
                }
            with
            | ex ->
                printfn "⚠️ GPU allocation failed: %s" ex.Message
                {
                    Data = data
                    Shape = shape
                    Device = None
                    IsOnGPU = false
                }
        | None ->
            {
                Data = data
                Shape = shape
                Device = None
                IsOnGPU = false
            }

    /// Matrix multiplication with CUDA acceleration
    let matrixMultiply (a: TarsTensor<float32>) (b: TarsTensor<float32>) (cudaContext: CudaContext option) : TarsTensor<float32> =
        if a.Shape.Length <> 2 || b.Shape.Length <> 2 then
            failwith "Matrix multiplication requires 2D tensors"
        
        let [|m; k|] = a.Shape
        let [|k2; n|] = b.Shape
        
        if k <> k2 then
            failwith $"Matrix dimensions don't match: {k} vs {k2}"
        
        let resultShape = [|m; n|]
        let resultSize = m * n
        
        match cudaContext, a.Device, b.Device with
        | Some ctx, Some deviceA, Some deviceB when a.IsOnGPU && b.IsOnGPU ->
            try
                // GPU matrix multiplication using ILGPU
                let resultBuffer = ctx.Accelerator.Allocate1D<float32>(resultSize)
                
                // Simple GPU kernel for matrix multiplication
                let kernel = ctx.Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, 
                    ArrayView1D<float32, Stride1D.Dense>,
                    ArrayView1D<float32, Stride1D.Dense>,
                    ArrayView1D<float32, Stride1D.Dense>,
                    int, int, int>(fun index a b result m k n ->
                    let row = index / n
                    let col = index % n
                    let mutable sum = 0.0f
                    for i = 0 to k - 1 do
                        sum <- sum + a.[row * k + i] * b.[i * n + col]
                    result.[index] <- sum
                )
                
                kernel.Invoke(ctx.Stream, resultSize, deviceA.View, deviceB.View, resultBuffer.View, m, k, n)
                ctx.Stream.Synchronize()
                
                let resultData = resultBuffer.GetAsArray1D()
                
                {
                    Data = resultData
                    Shape = resultShape
                    Device = Some(resultBuffer)
                    IsOnGPU = true
                }
            with
            | ex ->
                printfn "⚠️ GPU matrix multiplication failed: %s" ex.Message
                // Fallback to CPU
                matrixMultiplyCPU a b resultShape
        | _ ->
            // CPU matrix multiplication
            matrixMultiplyCPU a b resultShape

    /// CPU fallback for matrix multiplication
    and private matrixMultiplyCPU (a: TarsTensor<float32>) (b: TarsTensor<float32>) (resultShape: int[]) : TarsTensor<float32> =
        let [|m; k|] = a.Shape
        let [|_; n|] = b.Shape
        let result = Array.zeroCreate<float32> (m * n)
        
        for i = 0 to m - 1 do
            for j = 0 to n - 1 do
                let mutable sum = 0.0f
                for l = 0 to k - 1 do
                    sum <- sum + a.Data.[i * k + l] * b.Data.[l * n + j]
                result.[i * n + j] <- sum
        
        {
            Data = result
            Shape = resultShape
            Device = None
            IsOnGPU = false
        }

    /// Element-wise operations
    let elementWiseAdd (a: TarsTensor<float32>) (b: TarsTensor<float32>) (cudaContext: CudaContext option) : TarsTensor<float32> =
        if a.Shape <> b.Shape then
            failwith "Tensors must have the same shape for element-wise addition"
        
        match cudaContext, a.Device, b.Device with
        | Some ctx, Some deviceA, Some deviceB when a.IsOnGPU && b.IsOnGPU ->
            try
                let resultBuffer = ctx.Accelerator.Allocate1D<float32>(a.Data.Length)
                
                let kernel = ctx.Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView1D<float32, Stride1D.Dense>,
                    ArrayView1D<float32, Stride1D.Dense>,
                    ArrayView1D<float32, Stride1D.Dense>>(fun index a b result ->
                    result.[index] <- a.[index] + b.[index]
                )
                
                kernel.Invoke(ctx.Stream, a.Data.Length, deviceA.View, deviceB.View, resultBuffer.View)
                ctx.Stream.Synchronize()
                
                let resultData = resultBuffer.GetAsArray1D()
                
                {
                    Data = resultData
                    Shape = a.Shape
                    Device = Some(resultBuffer)
                    IsOnGPU = true
                }
            with
            | ex ->
                printfn "⚠️ GPU element-wise add failed: %s" ex.Message
                elementWiseAddCPU a b
        | _ ->
            elementWiseAddCPU a b

    /// CPU fallback for element-wise addition
    and private elementWiseAddCPU (a: TarsTensor<float32>) (b: TarsTensor<float32>) : TarsTensor<float32> =
        let result = Array.map2 (+) a.Data b.Data
        {
            Data = result
            Shape = a.Shape
            Device = None
            IsOnGPU = false
        }

    /// Activation functions
    let relu (tensor: TarsTensor<float32>) (cudaContext: CudaContext option) : TarsTensor<float32> =
        match cudaContext, tensor.Device with
        | Some ctx, Some device when tensor.IsOnGPU ->
            try
                let resultBuffer = ctx.Accelerator.Allocate1D<float32>(tensor.Data.Length)
                
                let kernel = ctx.Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView1D<float32, Stride1D.Dense>,
                    ArrayView1D<float32, Stride1D.Dense>>(fun index input result ->
                    result.[index] <- max 0.0f input.[index]
                )
                
                kernel.Invoke(ctx.Stream, tensor.Data.Length, device.View, resultBuffer.View)
                ctx.Stream.Synchronize()
                
                let resultData = resultBuffer.GetAsArray1D()
                
                {
                    Data = resultData
                    Shape = tensor.Shape
                    Device = Some(resultBuffer)
                    IsOnGPU = true
                }
            with
            | ex ->
                printfn "⚠️ GPU ReLU failed: %s" ex.Message
                reluCPU tensor
        | _ ->
            reluCPU tensor

    /// CPU fallback for ReLU
    and private reluCPU (tensor: TarsTensor<float32>) : TarsTensor<float32> =
        let result = Array.map (max 0.0f) tensor.Data
        {
            Data = result
            Shape = tensor.Shape
            Device = None
            IsOnGPU = false
        }

    /// Softmax activation
    let softmax (tensor: TarsTensor<float32>) : TarsTensor<float32> =
        // CPU implementation for now (GPU version would be more complex)
        let maxVal = Array.max tensor.Data
        let expValues = Array.map (fun x -> Math.Exp(float (x - maxVal)) |> float32) tensor.Data
        let sumExp = Array.sum expValues
        let result = Array.map (fun x -> x / sumExp) expValues
        
        {
            Data = result
            Shape = tensor.Shape
            Device = None
            IsOnGPU = false
        }

    /// Copy tensor from GPU to CPU
    let toCPU (tensor: TarsTensor<'T>) : TarsTensor<'T> =
        match tensor.Device with
        | Some device when tensor.IsOnGPU ->
            let cpuData = device.GetAsArray1D()
            {
                Data = cpuData
                Shape = tensor.Shape
                Device = None
                IsOnGPU = false
            }
        | _ -> tensor

    /// Dispose CUDA resources
    let disposeCuda (cudaContext: CudaContext) =
        try
            cudaContext.Stream.Dispose()
            cudaContext.Accelerator.Dispose()
            cudaContext.Context.Dispose()
            printfn "✅ CUDA resources disposed"
        with
        | ex ->
            printfn "⚠️ Error disposing CUDA resources: %s" ex.Message
