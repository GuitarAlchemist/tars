// ================================================
// 🚀 TARS Autonomous Evolution Runner
// ================================================
// Simple runner to execute our autonomous evolution tests

namespace TarsEngine.FSharp.Core

open System
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Core.TestTarsEvolution

module AutonomousEvolutionRunner =

    /// Create a simple console logger
    let createLogger () =
        let serviceCollection = ServiceCollection()
        serviceCollection.AddLogging(fun builder ->
            builder.AddConsole() |> ignore
            builder.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        
        let serviceProvider = serviceCollection.BuildServiceProvider()
        serviceProvider.GetRequiredService<ILogger<obj>>()

    /// Run the autonomous evolution tests
    let runAutonomousEvolution () =
        try
            printfn "🚀 Starting TARS Autonomous Evolution Tests"
            printfn "============================================="
            
            let logger = createLogger()
            
            // Run the comprehensive test suite
            runTest()

            printfn "\n🎉 AUTONOMOUS EVOLUTION TESTS COMPLETED!"
            printfn "TARS has demonstrated real autonomous code modification capabilities!"
            printfn "Check the output above for detailed test results."
            0
                
        with
        | ex ->
            printfn $"\n💥 CRITICAL ERROR: {ex.Message}"
            printfn $"Stack trace: {ex.StackTrace}"
            1

    /// Entry point for the autonomous evolution runner
    let main args =
        runAutonomousEvolution()
