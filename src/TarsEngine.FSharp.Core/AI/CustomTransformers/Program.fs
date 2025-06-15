open System
open TarsEngine.CustomTransformers.CudaHybridOperations

[<EntryPoint>]
let main argv =
    try
        printfn "🌌 TARS CUSTOM TRANSFORMERS - CUDA HYBRID OPERATIONS"
        printfn "===================================================="
        printfn ""

        // Test CUDA operations first
        let cudaSuccess = testCudaOperations()

        printfn ""
        printfn "%s" (String.replicate 50 "=")
        printfn ""

        // Run hybrid embeddings demo
        demoHybridEmbeddings()

        printfn ""
        printfn "🎉 TARS Custom Transformers CUDA operations completed!"
        printfn "CUDA operations successful: %b" cudaSuccess

        0 // Success
    with
    | ex ->
        printfn "❌ CUDA operations failed: %s" ex.Message
        1 // Error
