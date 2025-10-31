namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.Agents

/// <summary>
/// TARS CUDA Vector Store Command - Real GPU Acceleration
/// Demonstrates genuine CUDA-accelerated vector operations
/// </summary>
type CudaCommand(logger: ILogger<CudaCommand>) =
    
    interface ICommand with
        member _.Name = "cuda"
        member _.Description = "TARS CUDA Vector Store - Real GPU acceleration for semantic search"
        member _.Usage = "tars cuda [test|benchmark|status] [options]"
        
        member this.ExecuteAsync(args: string[]) (options: CommandOptions) =
            task {
                try
                    // Display TARS CUDA header
                    let rule = Rule("[bold green]🚀 TARS CUDA VECTOR STORE[/]")
                    rule.Justification <- Justify.Center
                    AnsiConsole.Write(rule)
                    AnsiConsole.WriteLine()
                    
                    match args with
                    | [||] | [|"help"|] ->
                        // Show help information
                        AnsiConsole.MarkupLine("[yellow]📖 TARS CUDA Commands:[/]")
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[cyan]  test[/]                      - Test CUDA vector store functionality")
                        AnsiConsole.MarkupLine("[cyan]  benchmark[/]                 - Run performance benchmarks")
                        AnsiConsole.MarkupLine("[cyan]  status[/]                    - Show CUDA system status")
                        AnsiConsole.MarkupLine("[cyan]  demo[/]                      - Run comprehensive demo")
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[dim]Examples:[/]")
                        AnsiConsole.MarkupLine("[dim]  tars cuda test[/]")
                        AnsiConsole.MarkupLine("[dim]  tars cuda benchmark[/]")
                        AnsiConsole.MarkupLine("[dim]  tars cuda status[/]")
                        return CommandResult.success "TARS CUDA help displayed"
                        
                    | [|"test"|] ->
                        // Test CUDA functionality
                        return! this.RunTest()
                        
                    | [|"benchmark"|] ->
                        // Run performance benchmarks
                        return! this.RunBenchmark()
                        
                    | [|"status"|] ->
                        // Show CUDA status
                        return! this.ShowStatus()
                        
                    | [|"demo"|] ->
                        // Run comprehensive demo
                        return! this.RunDemo()
                        
                    | _ ->
                        AnsiConsole.MarkupLine("[red]❌ Unknown CUDA command. Use 'tars cuda help' for usage.[/]")
                        return CommandResult.failure "Unknown CUDA command"
                        
                with ex ->
                    logger.LogError(ex, "Error in CudaCommand")
                    AnsiConsole.MarkupLine($"[red]❌ CUDA command failed: {ex.Message}[/]")
                    return CommandResult.failure $"CUDA command failed: {ex.Message}"
            }
    
    /// Test CUDA functionality
    member private this.RunTest() =
        task {
            AnsiConsole.MarkupLine("[blue]🧪 CUDA Vector Store Test[/]")
            AnsiConsole.WriteLine()
            
            // Check CUDA availability
            if RealCudaVectorStore.IsCudaAvailable() then
                AnsiConsole.MarkupLine("[green]✅ CUDA is available[/]")
                
                // Create test store
                let optimal = CudaVectorStoreFactory.GetOptimalParameters()
                match CudaVectorStoreFactory.CreateStore(1000, 128, 0, logger) with
                | Ok store ->
                    use store = store
                    
                    AnsiConsole.MarkupLine("[green]✅ CUDA Vector Store created successfully[/]")
                    AnsiConsole.MarkupLine($"[dim]  Max Vectors: {store.MaxVectors}[/]")
                    AnsiConsole.MarkupLine($"[dim]  Vector Dimension: {store.VectorDimension}[/]")
                    
                    // Generate test vectors
                    let testVectors = 
                        [| for i in 0..99 do
                            yield [| for j in 0..127 do yield float32 (i + j) * 0.01f |]
                        |]
                    
                    // Add vectors
                    match store.AddVectors(testVectors) with
                    | Ok count ->
                        AnsiConsole.MarkupLine($"[green]✅ Added {testVectors.Length} vectors (total: {count})[/]")
                        
                        // Test search
                        let query = [| for i in 0..127 do yield float32 i * 0.01f |]
                        match store.Search(query, 5) with
                        | Ok result ->
                            AnsiConsole.MarkupLine("[green]✅ Search completed successfully[/]")
                            AnsiConsole.MarkupLine($"[dim]  Search Time: {result.Metrics.SearchTimeMs:F3} ms[/]")
                            AnsiConsole.MarkupLine($"[dim]  Throughput: {result.Metrics.ThroughputSearchesPerSec:F0} searches/sec[/]")
                            
                            return CommandResult.success "CUDA test completed successfully"
                        | Error error ->
                            AnsiConsole.MarkupLine($"[red]❌ Search failed: {error}[/]")
                            return CommandResult.failure $"Search failed: {error}"
                    | Error error ->
                        AnsiConsole.MarkupLine($"[red]❌ Failed to add vectors: {error}[/]")
                        return CommandResult.failure $"Failed to add vectors: {error}"
                        
                | Error error ->
                    AnsiConsole.MarkupLine($"[red]❌ Failed to create CUDA store: {error}[/]")
                    return CommandResult.failure $"Failed to create CUDA store: {error}"
            else
                AnsiConsole.MarkupLine("[red]❌ CUDA is not available[/]")
                return CommandResult.failure "CUDA is not available"
        }
    
    /// Run performance benchmarks
    member private this.RunBenchmark() =
        task {
            AnsiConsole.MarkupLine("[blue]🏁 CUDA Performance Benchmark[/]")
            AnsiConsole.WriteLine()
            
            if not (RealCudaVectorStore.IsCudaAvailable()) then
                AnsiConsole.MarkupLine("[red]❌ CUDA is not available[/]")
                return CommandResult.failure "CUDA is not available"
            
            // Create benchmark store
            match CudaVectorStoreFactory.CreateStore(50000, 384, 0, logger) with
            | Ok store ->
                use store = store
                
                AnsiConsole.MarkupLine("[green]✅ CUDA Benchmark Store created[/]")
                
                // Generate benchmark data
                let numVectors = 10000
                let vectorDim = 384
                
                let! benchmarkData =
                    AnsiConsole.Progress()
                        .Columns([|
                            TaskDescriptionColumn() :> ProgressColumn
                            ProgressBarColumn() :> ProgressColumn
                            PercentageColumn() :> ProgressColumn
                        |])
                        .StartAsync(fun ctx ->
                            task {
                                let task = ctx.AddTask("[green]Generating benchmark data...[/]")
                                task.StartTask()

                                let rnd = Random(42)
                                let vectors =
                                    [| for i in 0..numVectors-1 do
                                        task.Increment(100.0 / float numVectors)
                                        yield [| for j in 0..vectorDim-1 do yield float32 (rnd.NextDouble() * 2.0 - 1.0) |]
                                    |]

                                task.StopTask()
                                return vectors
                            })
                
                // Add vectors to store
                match store.AddVectors(benchmarkData) with
                | Ok count ->
                    AnsiConsole.MarkupLine($"[green]✅ Added {count} vectors for benchmarking[/]")
                    
                    // Single query benchmark
                    let query = [| for i in 0..vectorDim-1 do yield float32 (Random().NextDouble() * 2.0 - 1.0) |]
                    
                    AnsiConsole.MarkupLine("[yellow]🔍 Single Query Benchmark[/]")
                    match store.Search(query, 10) with
                    | Ok result ->
                        let table = Table()
                        table.AddColumn(TableColumn("[bold]Metric[/]")) |> ignore
                        table.AddColumn(TableColumn("[bold]Value[/]")) |> ignore
                        
                        table.AddRow([|"Search Time"; $"{result.Metrics.SearchTimeMs:F3} ms"|]) |> ignore
                        table.AddRow([|"Throughput"; $"{result.Metrics.ThroughputSearchesPerSec:F0} searches/sec"|]) |> ignore
                        table.AddRow([|"GPU Memory"; $"{result.Metrics.GpuMemoryUsedMb:F2} MB"|]) |> ignore
                        table.AddRow([|"Vectors Processed"; $"{result.Metrics.VectorsProcessed:N0}"|]) |> ignore
                        
                        AnsiConsole.Write(table)
                        
                        // Performance analysis
                        AnsiConsole.WriteLine()
                        if result.Metrics.ThroughputSearchesPerSec >= 1000.0 then
                            AnsiConsole.MarkupLine("[green]✅ Excellent performance: >1K searches/second[/]")
                        elif result.Metrics.ThroughputSearchesPerSec >= 100.0 then
                            AnsiConsole.MarkupLine("[yellow]⚠️ Good performance: >100 searches/second[/]")
                        else
                            AnsiConsole.MarkupLine("[red]❌ Performance needs optimization[/]")
                        
                        return CommandResult.success "CUDA benchmark completed"
                    | Error error ->
                        AnsiConsole.MarkupLine($"[red]❌ Benchmark failed: {error}[/]")
                        return CommandResult.failure $"Benchmark failed: {error}"
                | Error error ->
                    AnsiConsole.MarkupLine($"[red]❌ Failed to add benchmark data: {error}[/]")
                    return CommandResult.failure $"Failed to add benchmark data: {error}"
                    
            | Error error ->
                AnsiConsole.MarkupLine($"[red]❌ Failed to create benchmark store: {error}[/]")
                return CommandResult.failure $"Failed to create benchmark store: {error}"
        }
    
    /// Show CUDA system status
    member private this.ShowStatus() =
        task {
            AnsiConsole.MarkupLine("[blue]📊 CUDA System Status[/]")
            AnsiConsole.WriteLine()
            
            let statusTable = Table()
            statusTable.AddColumn(TableColumn("[bold]Component[/]")) |> ignore
            statusTable.AddColumn(TableColumn("[bold]Status[/]")) |> ignore
            statusTable.AddColumn(TableColumn("[bold]Details[/]")) |> ignore
            
            // Check CUDA availability
            let cudaAvailable = RealCudaVectorStore.IsCudaAvailable()
            statusTable.AddRow([|"CUDA Runtime";
                                (if cudaAvailable then "[green]Available[/]" else "[red]Not Available[/]");
                                (if cudaAvailable then "GPU acceleration ready" else "No GPU detected")|]) |> ignore
            
            // Check native library
            let nativeLibExists = File.Exists("./TarsEngine.CUDA.VectorStore/tars_optimized_vector_store")
            statusTable.AddRow([|"Native Library";
                                (if nativeLibExists then "[green]Found[/]" else "[red]Missing[/]");
                                (if nativeLibExists then "CUDA kernels compiled" else "Compile CUDA kernels")|]) |> ignore
            
            // Get optimal parameters
            let optimal = CudaVectorStoreFactory.GetOptimalParameters()
            statusTable.AddRow([|"Optimal Config"; "[blue]Ready[/]"; 
                                $"Vectors: {optimal.MaxVectors:N0}, Dim: {optimal.VectorDim}"|]) |> ignore
            
            AnsiConsole.Write(statusTable)
            AnsiConsole.WriteLine()
            
            if cudaAvailable && nativeLibExists then
                AnsiConsole.MarkupLine("[green]✅ CUDA Vector Store fully operational[/]")
            else
                AnsiConsole.MarkupLine("[yellow]⚠️ CUDA Vector Store requires setup[/]")
            
            return CommandResult.success "CUDA status displayed"
        }
    
    /// Run comprehensive demo
    member private this.RunDemo() =
        task {
            AnsiConsole.MarkupLine("[blue]🎯 CUDA Vector Store Comprehensive Demo[/]")
            AnsiConsole.WriteLine()
            
            // Run all components
            let! testResult = this.RunTest()
            if testResult.ExitCode = 0 then
                AnsiConsole.WriteLine()
                let! benchmarkResult = this.RunBenchmark()
                if benchmarkResult.ExitCode = 0 then
                    AnsiConsole.WriteLine()
                    let! statusResult = this.ShowStatus()
                    return statusResult
                else
                    return benchmarkResult
            else
                return testResult
        }
