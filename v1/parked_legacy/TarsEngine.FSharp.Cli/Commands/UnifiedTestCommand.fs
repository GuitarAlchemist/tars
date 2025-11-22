namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Commands.Types

/// Unified Test Command - Runs comprehensive tests for all unified systems
module UnifiedTestCommand =
    
    /// Run unified system tests
    let runUnifiedTests (logger: ITarsLogger) (testFilter: string option) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🧪 TARS Unified System Test Suite[/]")
                AnsiConsole.MarkupLine("[dim]Comprehensive testing of all unified systems[/]")
                AnsiConsole.WriteLine()
                
                // Determine test project path
                let testProjectPath = "TarsEngine.FSharp.Tests/TarsEngine.FSharp.Tests.fsproj"
                
                if not (System.IO.File.Exists(testProjectPath)) then
                    AnsiConsole.MarkupLine("[red]❌ Test project not found at: TarsEngine.FSharp.Tests/[/]")
                    return 1
                
                AnsiConsole.MarkupLine("[yellow]🔧 Building test project...[/]")
                
                // Build test project
                let buildArgs = $"build \"{testProjectPath}\" --configuration Release --verbosity quiet"
                let buildProcess = Process.Start(ProcessStartInfo(
                    FileName = "dotnet",
                    Arguments = buildArgs,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                ))
                
                buildProcess.WaitForExit()
                
                if buildProcess.ExitCode <> 0 then
                    let error = buildProcess.StandardError.ReadToEnd()
                    AnsiConsole.MarkupLine($"[red]❌ Build failed: {error}[/]")
                    return 1
                
                AnsiConsole.MarkupLine("[green]✅ Test project built successfully[/]")
                AnsiConsole.WriteLine()
                
                // Prepare test arguments
                let testArgs = 
                    match testFilter with
                    | Some filter -> $"test \"{testProjectPath}\" --configuration Release --verbosity normal --filter \"{filter}\" --logger \"console;verbosity=detailed\""
                    | None -> $"test \"{testProjectPath}\" --configuration Release --verbosity normal --logger \"console;verbosity=detailed\""
                
                AnsiConsole.MarkupLine("[yellow]🧪 Running unified system tests...[/]")
                
                if testFilter.IsSome then
                    AnsiConsole.MarkupLine($"[dim]Filter: {testFilter.Value}[/]")
                
                AnsiConsole.WriteLine()
                
                // Run tests
                let testProcess = Process.Start(ProcessStartInfo(
                    FileName = "dotnet",
                    Arguments = testArgs,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                ))
                
                let output = testProcess.StandardOutput.ReadToEnd()
                let error = testProcess.StandardError.ReadToEnd()
                
                testProcess.WaitForExit()
                
                // Parse and display test results
                let lines = output.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                
                let mutable testsPassed = 0
                let mutable testsFailed = 0
                let mutable testsSkipped = 0
                let mutable totalTime = ""
                
                for line in lines do
                    if line.Contains("Passed!") then
                        AnsiConsole.MarkupLine($"[green]✅ {line.Trim()}[/]")
                    elif line.Contains("Failed!") then
                        AnsiConsole.MarkupLine($"[red]❌ {line.Trim()}[/]")
                    elif line.Contains("Skipped:") then
                        AnsiConsole.MarkupLine($"[yellow]⏭️ {line.Trim()}[/]")
                    elif line.Contains("Total tests:") then
                        AnsiConsole.MarkupLine($"[cyan]📊 {line.Trim()}[/]")
                    elif line.Contains("Passed:") && line.Contains("Failed:") then
                        AnsiConsole.MarkupLine($"[bold cyan]📈 {line.Trim()}[/]")
                        
                        // Extract numbers
                        let parts = line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
                        for i in 0..parts.Length-2 do
                            if parts.[i] = "Passed:" && i+1 < parts.Length then
                                Int32.TryParse(parts.[i+1].TrimEnd(','), &testsPassed) |> ignore
                            elif parts.[i] = "Failed:" && i+1 < parts.Length then
                                Int32.TryParse(parts.[i+1].TrimEnd(','), &testsFailed) |> ignore
                            elif parts.[i] = "Skipped:" && i+1 < parts.Length then
                                Int32.TryParse(parts.[i+1].TrimEnd(','), &testsSkipped) |> ignore
                    elif line.Contains("Time:") then
                        totalTime <- line.Trim()
                        AnsiConsole.MarkupLine($"[dim]⏱️ {totalTime}[/]")
                
                AnsiConsole.WriteLine()
                
                // Show detailed results
                AnsiConsole.MarkupLine("[bold cyan]📊 Test Results Summary:[/]")
                AnsiConsole.MarkupLine($"  Tests Passed: [green]{testsPassed}[/]")
                AnsiConsole.MarkupLine($"  Tests Failed: [red]{testsFailed}[/]")
                AnsiConsole.MarkupLine($"  Tests Skipped: [yellow]{testsSkipped}[/]")
                
                let totalTests = testsPassed + testsFailed + testsSkipped
                if totalTests > 0 then
                    let successRate = (float testsPassed / float totalTests) * 100.0
                    let successRateStr = successRate.ToString("F1")
                    AnsiConsole.MarkupLine($"  Success Rate: [yellow]{successRateStr}%[/]")
                
                if not (String.IsNullOrEmpty(totalTime)) then
                    AnsiConsole.MarkupLine($"  {totalTime}")
                
                AnsiConsole.WriteLine()
                
                // Show any errors
                if not (String.IsNullOrEmpty(error)) && error.Trim().Length > 0 then
                    AnsiConsole.MarkupLine("[yellow]⚠️ Test Warnings/Errors:[/]")
                    let errorLines = error.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                    for errorLine in errorLines do
                        if not (errorLine.Trim().StartsWith("Determining projects")) then
                            AnsiConsole.MarkupLine($"[dim]{errorLine.Trim()}[/]")
                    AnsiConsole.WriteLine()
                
                // Final status
                if testProcess.ExitCode = 0 && testsFailed = 0 then
                    AnsiConsole.MarkupLine("[bold green]🎉 All Unified System Tests Passed![/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold cyan]🚀 UNIFIED SYSTEM TEST ACHIEVEMENTS:[/]")
                    AnsiConsole.MarkupLine("  ✅ [green]Core System Tests[/] - Unified types and error handling")
                    AnsiConsole.MarkupLine("  ✅ [green]Configuration Tests[/] - Centralized configuration management")
                    AnsiConsole.MarkupLine("  ✅ [green]Proof System Tests[/] - Cryptographic evidence generation")
                    AnsiConsole.MarkupLine("  ✅ [green]CUDA Engine Tests[/] - GPU acceleration and fallback")
                    AnsiConsole.MarkupLine("  ✅ [green]Integration Tests[/] - Cross-system communication")
                    AnsiConsole.MarkupLine("  ✅ [green]Performance Tests[/] - System performance validation")
                    return 0
                else
                    AnsiConsole.MarkupLine("[red]❌ Some tests failed or encountered errors[/]")
                    return 1
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Test execution failed: {ex.Message}[/]")
                return 1
        }
    
    /// Show test categories and available filters
    let showTestCategories (logger: ITarsLogger) =
        AnsiConsole.MarkupLine("[bold cyan]🧪 TARS Unified System Test Categories[/]")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("[bold yellow]Available Test Categories:[/]")
        AnsiConsole.MarkupLine("  [cyan]Core[/] - Unified core system tests")
        AnsiConsole.MarkupLine("    • UnifiedCoreTests - Core types and error handling")
        AnsiConsole.MarkupLine("    • UnifiedLoggerTests - Centralized logging system")
        AnsiConsole.MarkupLine("    • UnifiedStateManagerTests - Thread-safe state management")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("  [cyan]Configuration[/] - Configuration management tests")
        AnsiConsole.MarkupLine("    • UnifiedConfigurationManagerTests - Centralized configuration")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("  [cyan]Proof[/] - Cryptographic proof system tests")
        AnsiConsole.MarkupLine("    • UnifiedProofSystemTests - Proof generation and verification")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("  [cyan]CUDA[/] - GPU acceleration tests")
        AnsiConsole.MarkupLine("    • UnifiedCudaEngineTests - GPU operations and fallback")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("  [cyan]Integration[/] - Cross-system integration tests")
        AnsiConsole.MarkupLine("    • UnifiedIntegrationTests - End-to-end system testing")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("  [cyan]Performance[/] - Performance and load tests")
        AnsiConsole.MarkupLine("    • UnifiedPerformanceTests - System performance validation")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("  [cyan]Security[/] - Security and validation tests")
        AnsiConsole.MarkupLine("    • UnifiedSecurityTests - Security validation")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("[bold yellow]Example Filters:[/]")
        AnsiConsole.MarkupLine("  [dim]tars test --filter Core[/]                    # Run only core tests")
        AnsiConsole.MarkupLine("  [dim]tars test --filter Configuration[/]           # Run only configuration tests")
        AnsiConsole.MarkupLine("  [dim]tars test --filter Integration[/]             # Run only integration tests")
        AnsiConsole.MarkupLine("  [dim]tars test --filter \"Core|Configuration\"[/]    # Run core and configuration tests")
        AnsiConsole.WriteLine()
    
    /// Unified Test Command implementation
    type UnifiedTestCommand() =
        interface ICommand with
            member _.Name = "test"
            member _.Description = "Run comprehensive tests for all unified TARS systems"
            member _.Usage = "tars test [--filter category] [--list]"
            member _.Examples = [
                "tars test                     # Run all unified system tests"
                "tars test --filter Core       # Run only core system tests"
                "tars test --filter Integration # Run only integration tests"
                "tars test --list              # Show available test categories"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedTestCommand"
                        
                        let isListMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--list")
                        
                        let filterArg = 
                            options.Arguments 
                            |> List.tryFind (fun arg -> arg.StartsWith("--filter"))
                        
                        let testFilter = 
                            match filterArg with
                            | Some filter when filter.Contains("=") ->
                                let parts = filter.Split('=', 2)
                                if parts.Length = 2 then Some parts.[1] else None
                            | Some _ ->
                                // Look for next argument as filter value
                                match options.Arguments |> List.tryFindIndex (fun arg -> arg = "--filter") with
                                | Some index when index + 1 < options.Arguments.Length ->
                                    Some options.Arguments.[index + 1]
                                | _ -> None
                            | None -> None
                        
                        if isListMode then
                            showTestCategories logger
                            return { Message = ""; ExitCode = 0; Success = true }
                        else
                            let! result = runUnifiedTests logger testFilter
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[red]❌ Test command failed: {ex.Message}[/]")
                        return { Message = ""; ExitCode = 1; Success = false }
                }

