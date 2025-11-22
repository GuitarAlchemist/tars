namespace TarsEngine

open System
open System.IO
open System.Reflection
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Configuration
open TarsEngine.CudaTestConsole

/// TARS Engine Main Program - Real GPU Execution Entry Point
module Program =
    
    /// Configure comprehensive logging with multiple providers
    let configureLogging (builder: ILoggingBuilder) =
        builder
            .ClearProviders()
            .AddConsole(fun options ->
                options.IncludeScopes <- true
                options.TimestampFormat <- "[yyyy-MM-dd HH:mm:ss.fff] "
                options.DisableColors <- false)
            .AddDebug()
            .SetMinimumLevel(LogLevel.Information)
        |> ignore
    
    /// Configure comprehensive services
    let configureServices (services: IServiceCollection) =
        // Core services
        services.AddLogging(configureLogging) |> ignore
        services.AddSingleton<IConfiguration>(fun _ ->
            ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional = true)
                .AddEnvironmentVariables()
                .AddCommandLine(Environment.GetCommandLineArgs())
                .Build()) |> ignore
        
        // CUDA test services
        services.AddTransient<CudaKernelTest.CudaKernelTestSuite>() |> ignore
        services.AddTransient<CudaMemoryTests.CudaMemoryTestSuite>() |> ignore
        services.AddTransient<CudaPerformanceTests.CudaPerformanceTestSuite>() |> ignore
        services.AddTransient<CudaAdvancedKernelTests.CudaAdvancedKernelTestSuite>() |> ignore
        services.AddTransient<CudaErrorHandlingTests.CudaErrorHandlingTestSuite>() |> ignore
        services.AddTransient<CudaComprehensiveTestRunner.CudaComprehensiveTestRunner>() |> ignore
        
        services
    
    /// Print application banner
    let printBanner() =
        let version = Assembly.GetExecutingAssembly().GetName().Version
        let buildDate = File.GetLastWriteTime(Assembly.GetExecutingAssembly().Location)

        Console.ForegroundColor <- ConsoleColor.Cyan
        printfn ""
        printfn "████████╗ █████╗ ██████╗ ███████╗    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗"
        printfn "╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝"
        printfn "   ██║   ███████║██████╔╝███████╗    █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  "
        printfn "   ██║   ██╔══██║██╔══██╗╚════██║    ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  "
        printfn "   ██║   ██║  ██║██║  ██║███████║    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗"
        printfn "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝"
        Console.ForegroundColor <- ConsoleColor.Yellow
        printfn ""
        printfn "                    🚀 CUDA COMPREHENSIVE TEST SUITE 🚀"
        printfn "                         Real GPU Execution - No Simulations!"
        Console.ForegroundColor <- ConsoleColor.White
        printfn ""
        printfn "Version: %A" version
        printfn "Build Date: %s" (buildDate.ToString("yyyy-MM-dd HH:mm:ss"))
        printfn "Runtime: %s" System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription
        printfn "Platform: %s" System.Runtime.InteropServices.RuntimeInformation.OSDescription
        printfn "Architecture: %A" System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture
        Console.ResetColor()
        printfn ""
    
    /// Print usage information
    let printUsage() =
        printfn "TARS CUDA Test Engine - Usage"
        printfn "============================="
        printfn ""
        printfn "Usage: TarsEngine [command] [options]"
        printfn ""
        printfn "Commands:"
        printfn "  test [mode]        Run CUDA tests"
        printfn "    basic            - Run basic kernel tests only"
        printfn "    memory           - Run memory management tests only"
        printfn "    performance      - Run performance benchmark tests only"
        printfn "    advanced         - Run advanced kernel tests only"
        printfn "    errors           - Run error handling tests only"
        printfn "    comprehensive    - Run all tests (default)"
        printfn "    all              - Same as comprehensive"
        printfn ""
        printfn "  benchmark          Run performance benchmarks"
        printfn "  diagnose           Run system diagnostics"
        printfn "  version            Show version information"
        printfn "  help               Show this help message"
        printfn ""
    
    /// Parse command line arguments
    let parseArguments (args: string[]) =
        let mutable command = "test"
        let mutable testMode = "comprehensive"

        if args.Length > 0 then
            match args.[0] with
            | "test" ->
                command <- "test"
                if args.Length > 1 then testMode <- args.[1]
            | "benchmark" -> command <- "benchmark"
            | "diagnose" -> command <- "diagnose"
            | "version" -> command <- "version"
            | "help" -> command <- "help"
            | _ -> testMode <- args.[0]

        {| Command = command; TestMode = testMode |}
    
    /// Run system diagnostics
    let runDiagnostics() = async {
        printfn "🔍 Running TARS system diagnostics..."

        // Check .NET runtime
        let runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription
        printfn "✅ .NET Runtime: %s" runtime

        // Check CUDA availability
        try
            let deviceCount = CudaInterop.tars_cuda_device_count()
            if deviceCount > 0 then
                printfn "✅ CUDA Devices: %d found" deviceCount
            else
                printfn "⚠️ No CUDA devices found"
        with
        | ex -> printfn "❌ CUDA check failed: %s" ex.Message

        // Check CUDA library
        let cudaLibPath = Path.Combine(Directory.GetCurrentDirectory(), "libTarsCudaKernels.so")
        if File.Exists(cudaLibPath) then
            printfn "✅ CUDA Library: Found"
        else
            printfn "⚠️ CUDA Library: Not found"

        printfn "🏁 Diagnostics complete"
        return 0
    }
    
    /// Main entry point
    [<EntryPoint>]
    let main args =
        try
            printBanner()

            let parsedArgs = parseArguments args

            // Handle special commands
            match parsedArgs.Command with
            | "help" ->
                printUsage()
                0
            | "version" ->
                let version = Assembly.GetExecutingAssembly().GetName().Version
                printfn "TARS Engine Version %A" version
                0
            | "diagnose" ->
                runDiagnostics() |> Async.RunSynchronously
            | "benchmark" ->
                printfn "🚀 Running CUDA benchmarks..."
                runCudaTests [| "performance" |]
            | "test" ->
                let testArgs = [| parsedArgs.TestMode |]
                runCudaTests testArgs
            | _ ->
                printfn "Unknown command: %s" parsedArgs.Command
                printUsage()
                3
        with
        | ex ->
            printfn "💥 Fatal error: %s" ex.Message
            2
