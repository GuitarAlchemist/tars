namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Acceleration.UnifiedCudaEngine
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Core.UnifiedTypes

/// Unified CUDA Command - Demonstrates GPU acceleration capabilities
module UnifiedCudaCommand =
    
    /// Demonstrate the unified CUDA engine
    let demonstrateCudaEngine (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]⚡ TARS Unified CUDA Engine Demo[/]")
                AnsiConsole.MarkupLine("[dim]GPU acceleration for high-performance computing[/]")
                AnsiConsole.WriteLine()
                
                // Create CUDA engine
                use cudaEngine = createCudaEngine logger
                
                AnsiConsole.MarkupLine("[yellow]🔧 Initializing CUDA Engine...[/]")
                let! initResult = cudaEngine.InitializeAsync(CancellationToken.None)
                
                match initResult with
                | Failure (error, corrId) ->
                    AnsiConsole.MarkupLine($"[red]❌ CUDA initialization failed: {TarsError.toString error}[/]")
                    return 1
                | Success (_, metadata) ->
                    let deviceCount = metadata.["deviceCount"] :?> int
                    let isFallback = metadata.ContainsKey("fallbackMode")

                    if isFallback then
                        AnsiConsole.MarkupLine("[yellow]⚠️ No CUDA devices found - running in CPU fallback mode[/]")
                    else
                        let currentDevice = metadata.["currentDevice"] :?> int
                        AnsiConsole.MarkupLine($"[green]✅ CUDA engine initialized with {deviceCount} device(s)[/]")
                        AnsiConsole.MarkupLine($"[dim]Current device: {currentDevice}[/]")

                    // Show available devices
                    let devices = cudaEngine.GetAvailableDevices()
                    if devices.Length > 0 then
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[yellow]🖥️ Available CUDA Devices:[/]")

                        for device in devices do
                            AnsiConsole.MarkupLine($"  [cyan]Device {device.DeviceId}[/]: {device.Name}")
                            AnsiConsole.MarkupLine($"    [dim]Memory: {device.TotalMemory / 1024L / 1024L / 1024L} GB[/]")
                            let computeCapStr = device.ComputeCapability.ToString("F1")
                            AnsiConsole.MarkupLine($"    [dim]Compute Capability: {computeCapStr}[/]")
                            AnsiConsole.MarkupLine($"    [dim]Available: {device.IsAvailable}[/]")

                    AnsiConsole.WriteLine()

                    // Demonstrate various CUDA operations
                    AnsiConsole.MarkupLine("[yellow]🚀 Executing CUDA Operations...[/]")

                    let operations = [
                        // Vector similarity operations
                        ("Vector Similarity (512D)", CudaOperationFactory.createVectorSimilarity 512, "vector_data_512")
                        ("Vector Similarity (1024D)", CudaOperationFactory.createVectorSimilarity 1024, "vector_data_1024")
                        ("Vector Similarity (2048D)", CudaOperationFactory.createVectorSimilarity 2048, "vector_data_2048")

                        // Matrix multiplication operations
                        ("Matrix Multiply (256x256)", CudaOperationFactory.createMatrixMultiplication 256 256 256, "matrix_data_256")
                        ("Matrix Multiply (512x512)", CudaOperationFactory.createMatrixMultiplication 512 512 512, "matrix_data_512")
                        ("Matrix Multiply (1024x1024)", CudaOperationFactory.createMatrixMultiplication 1024 1024 1024, "matrix_data_1024")

                        // Reasoning kernel operations
                        ("Reasoning Kernel (Low)", CudaOperationFactory.createReasoningKernel 100, "reasoning_data_low")
                        ("Reasoning Kernel (Medium)", CudaOperationFactory.createReasoningKernel 500, "reasoning_data_medium")
                        ("Reasoning Kernel (High)", CudaOperationFactory.createReasoningKernel 1000, "reasoning_data_high")
                    ]

                    let operationResults = ResizeArray<string * CudaOperationResult>()

                    for (description, operation, data) in operations do
                        AnsiConsole.MarkupLine($"  🔄 [dim]{description}[/]")

                        let! result = cudaEngine.ExecuteOperationAsync(operation, box data, CancellationToken.None)

                        match result with
                        | Success (cudaResult, metadata) ->
                            operationResults.Add(description, cudaResult)

                            let statusIcon = if cudaResult.Success then "✅" else "❌"
                            let statusColor = if cudaResult.Success then "green" else "red"

                            AnsiConsole.MarkupLine($"    {statusIcon} [{statusColor}]{description}[/]")
                            let timeStr = cudaResult.ExecutionTime.TotalMilliseconds.ToString("F1")
                            let memoryMB = cudaResult.MemoryUsed / 1024L / 1024L
                            let throughputStr = cudaResult.ThroughputGFlops.ToString("F2")
                            AnsiConsole.MarkupLine($"      [dim]Time: {timeStr}ms[/]")
                            AnsiConsole.MarkupLine($"      [dim]Memory: {memoryMB} MB[/]")
                            AnsiConsole.MarkupLine($"      [dim]Throughput: {throughputStr} GFLOPS[/]")

                            if metadata.ContainsKey("fallback") then
                                AnsiConsole.MarkupLine($"      [yellow]⚠️ CPU Fallback[/]")
                            elif metadata.ContainsKey("device") then
                                let device = metadata.["device"] :?> int
                                AnsiConsole.MarkupLine($"      [cyan]🖥️ GPU {device}[/]")

                            if cudaResult.ErrorMessage.IsSome then
                                AnsiConsole.MarkupLine($"      [red]Error: {cudaResult.ErrorMessage.Value}[/]")

                        | Failure (error, corrId) ->
                            operationResults.Add(description, {
                                OperationId = operation.OperationId
                                Success = false
                                ExecutionTime = TimeSpan.Zero
                                MemoryUsed = 0L
                                ThroughputGFlops = 0.0
                                ErrorMessage = Some (TarsError.toString error)
                                ResultData = None
                            })

                            AnsiConsole.MarkupLine($"    ❌ [red]{description}: {TarsError.toString error}[/]")

                    AnsiConsole.WriteLine()

                    // Show performance metrics
                    let metrics = cudaEngine.GetPerformanceMetrics()
                    AnsiConsole.MarkupLine("[bold cyan]📊 CUDA Performance Metrics:[/]")
                    AnsiConsole.MarkupLine($"  Total Operations: [yellow]{metrics.TotalOperations}[/]")
                    AnsiConsole.MarkupLine($"  Successful: [green]{metrics.SuccessfulOperations}[/]")
                    AnsiConsole.MarkupLine($"  Failed: [red]{metrics.FailedOperations}[/]")

                    let successRate = if metrics.TotalOperations > 0L then (float metrics.SuccessfulOperations / float metrics.TotalOperations) * 100.0 else 0.0
                    let successRateStr = successRate.ToString("F1")
                    AnsiConsole.MarkupLine($"  Success Rate: [yellow]{successRateStr}%[/]")

                    if metrics.TotalOperations > 0L then
                        let avgTime = metrics.TotalGpuTime.TotalMilliseconds / float metrics.TotalOperations
                        let avgTimeStr = avgTime.ToString("F1")
                        AnsiConsole.MarkupLine($"  Average Execution Time: [cyan]{avgTimeStr}ms[/]")

                    AnsiConsole.WriteLine()

                    // Performance analysis
                    AnsiConsole.MarkupLine("[yellow]📈 Performance Analysis:[/]")

                    let successfulResults = operationResults |> Seq.filter (fun (_, result) -> result.Success) |> Seq.toList

                    if successfulResults.Length > 0 then
                        let totalTime = successfulResults |> List.sumBy (fun (_, result) -> result.ExecutionTime.TotalMilliseconds)
                        let totalGFlops = successfulResults |> List.sumBy (fun (_, result) -> result.ThroughputGFlops)
                        let totalMemory = successfulResults |> List.sumBy (fun (_, result) -> result.MemoryUsed)

                        let totalTimeStr = totalTime.ToString("F1")
                        let totalGFlopsStr = totalGFlops.ToString("F2")
                        AnsiConsole.MarkupLine($"  Total Execution Time: [cyan]{totalTimeStr}ms[/]")
                        AnsiConsole.MarkupLine($"  Total Throughput: [yellow]{totalGFlopsStr} GFLOPS[/]")
                        AnsiConsole.MarkupLine($"  Total Memory Used: [magenta]{totalMemory / 1024L / 1024L} MB[/]")

                        // Find fastest and slowest operations
                        let fastestOp = successfulResults |> List.minBy (fun (_, result) -> result.ExecutionTime.TotalMilliseconds)
                        let slowestOp = successfulResults |> List.maxBy (fun (_, result) -> result.ExecutionTime.TotalMilliseconds)

                        let fastestTime = (snd fastestOp).ExecutionTime.TotalMilliseconds.ToString("F1")
                        let slowestTime = (snd slowestOp).ExecutionTime.TotalMilliseconds.ToString("F1")
                        AnsiConsole.MarkupLine($"  Fastest Operation: [green]{fst fastestOp}[/] ({fastestTime}ms)")
                        AnsiConsole.MarkupLine($"  Slowest Operation: [red]{fst slowestOp}[/] ({slowestTime}ms)")

                        // Performance categories
                        let vectorOps = successfulResults |> List.filter (fun (desc, _) -> desc.Contains("Vector"))
                        let matrixOps = successfulResults |> List.filter (fun (desc, _) -> desc.Contains("Matrix"))
                        let reasoningOps = successfulResults |> List.filter (fun (desc, _) -> desc.Contains("Reasoning"))

                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[yellow]🎯 Operation Categories:[/]")

                        if vectorOps.Length > 0 then
                            let avgVectorTime = vectorOps |> List.averageBy (fun (_, result) -> result.ExecutionTime.TotalMilliseconds)
                            let avgVectorTimeStr = avgVectorTime.ToString("F1")
                            AnsiConsole.MarkupLine($"  Vector Operations: [cyan]{vectorOps.Length}[/] (avg: {avgVectorTimeStr}ms)")

                        if matrixOps.Length > 0 then
                            let avgMatrixTime = matrixOps |> List.averageBy (fun (_, result) -> result.ExecutionTime.TotalMilliseconds)
                            let avgMatrixTimeStr = avgMatrixTime.ToString("F1")
                            AnsiConsole.MarkupLine($"  Matrix Operations: [blue]{matrixOps.Length}[/] (avg: {avgMatrixTimeStr}ms)")

                        if reasoningOps.Length > 0 then
                            let avgReasoningTime = reasoningOps |> List.averageBy (fun (_, result) -> result.ExecutionTime.TotalMilliseconds)
                            let avgReasoningTimeStr = avgReasoningTime.ToString("F1")
                            AnsiConsole.MarkupLine($"  Reasoning Operations: [magenta]{reasoningOps.Length}[/] (avg: {avgReasoningTimeStr}ms)")

                    AnsiConsole.WriteLine()

                    // Show active operations (should be empty by now)
                    let activeOps = cudaEngine.GetActiveOperations()
                    AnsiConsole.MarkupLine($"[yellow]🔄 Active Operations: {activeOps.Length}[/]")

                    // Cleanup
                    let! cleanupResult = cudaEngine.CleanupAsync(CancellationToken.None)
                    match cleanupResult with
                    | Success _ ->
                        AnsiConsole.MarkupLine("[green]✅ CUDA engine cleaned up successfully[/]")
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine($"[red]⚠️ Cleanup warning: {TarsError.toString error}[/]")

                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold green]🎉 Unified CUDA Engine Demo Completed Successfully![/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold cyan]🚀 CUDA ENGINE ACHIEVEMENTS:[/]")
                    AnsiConsole.MarkupLine("  ✅ [green]GPU device detection[/] - Automatic CUDA device discovery")
                    AnsiConsole.MarkupLine("  ✅ [green]Operation execution[/] - Vector, matrix, and reasoning operations")
                    AnsiConsole.MarkupLine("  ✅ [green]CPU fallback[/] - Automatic fallback when GPU unavailable")
                    AnsiConsole.MarkupLine("  ✅ [green]Performance monitoring[/] - Real-time metrics and analysis")
                    AnsiConsole.MarkupLine("  ✅ [green]Memory management[/] - Efficient GPU memory utilization")
                    AnsiConsole.MarkupLine("  ✅ [green]Unified integration[/] - Seamless integration with TARS architecture")

                    return 0
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ CUDA engine demo failed: {ex.Message}[/]")
                return 1
        }
    
    /// Unified CUDA Command implementation
    type UnifiedCudaCommand() =
        interface ICommand with
            member _.Name = "cuda"
            member _.Description = "Demonstrate TARS unified CUDA GPU acceleration engine"
            member _.Usage = "tars cuda [--demo]"
            member _.Examples = [
                "tars cuda --demo          # Run CUDA engine demonstration"
                "tars cuda                 # Show CUDA system overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedCudaCommand"
                        
                        let isDemoMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--demo")
                        
                        if isDemoMode then
                            let! result = demonstrateCudaEngine logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]⚡ TARS Unified CUDA Engine[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("This system provides GPU acceleration for high-performance")
                            AnsiConsole.MarkupLine("computing operations across all TARS modules.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Supported Operations:[/]")
                            AnsiConsole.MarkupLine("  🔢 [cyan]Vector Similarity[/] - High-dimensional vector operations")
                            AnsiConsole.MarkupLine("  📊 [blue]Matrix Multiplication[/] - Large-scale linear algebra")
                            AnsiConsole.MarkupLine("  🧠 [magenta]Reasoning Kernels[/] - AI reasoning acceleration")
                            AnsiConsole.MarkupLine("  📈 [green]Tensor Operations[/] - Multi-dimensional array processing")
                            AnsiConsole.MarkupLine("  🔄 [yellow]Data Processing[/] - Parallel data transformation")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Features:[/]")
                            AnsiConsole.MarkupLine("  ⚡ [green]GPU Acceleration[/] - CUDA-powered high performance")
                            AnsiConsole.MarkupLine("  🔄 [blue]CPU Fallback[/] - Automatic fallback when GPU unavailable")
                            AnsiConsole.MarkupLine("  📊 [cyan]Performance Monitoring[/] - Real-time metrics and analysis")
                            AnsiConsole.MarkupLine("  🧠 [magenta]Memory Management[/] - Efficient GPU memory utilization")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Available options:")
                            AnsiConsole.MarkupLine("  [yellow]--demo[/]  Run CUDA engine demonstration")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Example: [dim]tars cuda --demo[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[red]❌ Command failed: {ex.Message}[/]")
                        return { Message = ""; ExitCode = 1; Success = false }
                }

