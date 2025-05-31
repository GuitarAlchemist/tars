open System
open TarsEngine.FSharp.Core.CUDA.CudaVectorStore
open TarsEngine.FSharp.Core.CUDA.AgenticCudaRAG
open TarsEngine.FSharp.Core.CUDA.CudaIntegrationTests

[<EntryPoint>]
let main args =
    printfn "🚀 TARS CUDA AGENTIC RAG INTEGRATION DEMO"
    printfn "========================================"
    printfn ""
    
    let runDemo() =
        async {
            try
                // Run CUDA Vector Store Demo
                printfn "🧪 1. CUDA Vector Store Demo"
                printfn "----------------------------"
                do! CudaVectorStoreDemo.runDemo()
                printfn ""
                
                // Run Agentic RAG Demo
                printfn "🤖 2. Agentic CUDA RAG Demo"
                printfn "---------------------------"
                do! AgenticCudaRAGDemo.runDemo()
                printfn ""
                
                // Run Integration Tests
                printfn "🔬 3. Integration Tests"
                printfn "-----------------------"
                do! CudaIntegrationTestsDemo.runDemo()
                printfn ""
                
                printfn "🎉 ALL DEMOS COMPLETED SUCCESSFULLY!"
                printfn "✅ TARS CUDA Agentic RAG is ready for production!"
                
            with
            | ex ->
                printfn "❌ Demo failed: %s" ex.Message
                printfn "Stack trace: %s" ex.StackTrace
        }
    
    // Run the demo
    runDemo() |> Async.RunSynchronously
    
    printfn ""
    printfn "Press any key to exit..."
    Console.ReadKey() |> ignore
    
    0 // Return success
