open System
open TarsEngine.FSharp.Core.CUDA.CudaIntegration

[<EntryPoint>]
let main args =
    printfn "🚀 TARS CUDA INTEGRATION DEMO"
    printfn "=============================="
    printfn ""

    let runDemo() =
        async {
            try
                // Run CUDA Integration Demo
                printfn "🧪 CUDA Integration Demo"
                printfn "------------------------"
                do! runTarsCudaDemo()
                printfn ""

                printfn "🎉 DEMO COMPLETED SUCCESSFULLY!"
                printfn "✅ TARS CUDA Integration is working!"

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
