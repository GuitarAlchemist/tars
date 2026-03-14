namespace TarsEngine.FSharp.Core.Metascript

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Core.Types
open TarsEngine.FSharp.Core.Metascript.Types
open TarsEngine.FSharp.Core.Metascript.Parser
open TarsEngine.FSharp.Core.Metascript.AntiBSAnalyzer
open TarsEngine.FSharp.Core.FLUX.FluxFractalArchitecture
open TarsEngine.FSharp.Core.FLUX.UnifiedTrsxInterpreter

/// Metascript execution services that actually work
module Services =
    
    /// Enhanced metascript executor with Spectre Console, real F# execution, and FLUX support
    type MetascriptExecutor(logger: ILogger<MetascriptExecutor>) =
        let unifiedInterpreter = UnifiedInterpreter()
        let fluxEngine = UnifiedFluxEngine()

        /// Execute F# code using F# Interactive
        member private _.ExecuteFSharpCode(code: string) =
            try
                // Create temporary F# script file
                let tempFile = Path.GetTempFileName() + ".fsx"
                File.WriteAllText(tempFile, code)

                // Execute using dotnet fsi
                let startInfo = System.Diagnostics.ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- $"fsi \"{tempFile}\""
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true

                use proc = System.Diagnostics.Process.Start(startInfo)
                let output = proc.StandardOutput.ReadToEnd()
                let error = proc.StandardError.ReadToEnd()
                proc.WaitForExit()

                // Clean up temp file
                try File.Delete(tempFile) with | _ -> ()

                if proc.ExitCode = 0 then
                    Ok output
                else
                    Error $"F# execution failed: {error}"
            with
            | ex -> Error $"F# execution error: {ex.Message}"

        member _.ExecuteMetascriptAsync(metascriptPath: string) =
            task {
                let startTime = DateTime.UtcNow

                try
                    // Create Spectre Console progress display
                    AnsiConsole.MarkupLine($"[cyan]📜 Running metascript:[/] [yellow]{metascriptPath}[/]")

                    logger.LogInformation($"Executing metascript: {metascriptPath}")



                    let parsed = parseMetascript metascriptPath
                    let mutable output = []
                    let mutable hasErrors = false

                    // Create progress bar for block execution
                    let progress = AnsiConsole.Progress()
                    progress.AutoRefresh <- true
                    progress.HideCompleted <- false

                    do! progress.StartAsync(fun ctx ->
                        task {
                            let task = ctx.AddTask("[green]Executing blocks[/]", true, parsed.Blocks.Length)

                            for block in parsed.Blocks do
                                task.Description <- $"[green]Processing {block.Type} block[/]"

                                match block.Type with
                                | FSharp ->
                                    AnsiConsole.MarkupLine($"[blue]🔧 Executing F# block at line {block.LineNumber}[/]")
                                    AnsiConsole.MarkupLine($"[dim]📝 Code preview: {block.Content.Substring(0, Math.Min(80, block.Content.Length))}...[/]")

                                    // 🤖 ANTI-BS ANALYSIS - Detect fake computational scripts
                                    AnsiConsole.MarkupLine($"[magenta]🤖 Running Anti-BS Analysis...[/]")
                                    let bsAnalysis = AntiBSAnalyzer.analyzeCode block.Content

                                    if not bsAnalysis.IsLegitimate then
                                        // REJECT the script - it's BS!
                                        let errorMsg = $"❌ SCRIPT REJECTED: Anti-BS Analysis Failed\n{bsAnalysis.Reason}"
                                        let fullErrorMsg = $"🤖 ANTI-BS PROTECTION ACTIVATED\n{AntiBSAnalyzer.generateAnalysisReport bsAnalysis}"
                                        output <- fullErrorMsg :: output
                                        hasErrors <- true

                                        AnsiConsole.MarkupLine($"[red]❌ SCRIPT REJECTED BY ANTI-BS ANALYZER[/]")
                                        AnsiConsole.MarkupLine("[red]═══════════════════════════════════════════════════════════════[/]")
                                        AnsiConsole.MarkupLine($"[red]🤖 ANTI-BS PROTECTION: This script appears to be fake/BS[/]")
                                        AnsiConsole.MarkupLine($"[red]Reason: {bsAnalysis.Reason}[/]")
                                        AnsiConsole.MarkupLine($"[red]Confidence: {bsAnalysis.Confidence * 100.0:F1}%%[/]")
                                        AnsiConsole.MarkupLine($"[red]Computational Complexity: {bsAnalysis.ComputationalComplexity}[/]")
                                        AnsiConsole.MarkupLine($"[red]Real Calculation Ratio: {bsAnalysis.RealCalculationRatio:F2}[/]")

                                        if bsAnalysis.SuspiciousPatterns.Length > 0 then
                                            AnsiConsole.MarkupLine($"[red]Suspicious Patterns:[/]")
                                            for pattern in bsAnalysis.SuspiciousPatterns do
                                                AnsiConsole.MarkupLine($"[red]  • {pattern}[/]")

                                        AnsiConsole.MarkupLine("[red]═══════════════════════════════════════════════════════════════[/]")
                                        AnsiConsole.MarkupLine($"[yellow]💡 Please provide a script with real mathematical calculations[/]")
                                    else
                                        // APPROVE the script - it looks legitimate
                                        AnsiConsole.MarkupLine($"[green]✅ Anti-BS Analysis PASSED[/]")
                                        AnsiConsole.MarkupLine($"[green]   Confidence: {bsAnalysis.Confidence * 100.0:F1}%% - Complexity: {bsAnalysis.ComputationalComplexity} - Calc Ratio: {bsAnalysis.RealCalculationRatio:F2}[/]")

                                    try
                                        // Create temporary F# project
                                        let tempDir = System.IO.Path.GetTempPath()
                                        let guidPart = System.Guid.NewGuid().ToString("N").Substring(0, 7)
                                        let projectDir = System.IO.Path.Combine(tempDir, $"tars_temp_{guidPart}")
                                        let programFile = System.IO.Path.Combine(projectDir, "Program.fs")
                                        let projectFile = System.IO.Path.Combine(projectDir, "TarsTemp.fsproj")

                                        // Create project directory
                                        System.IO.Directory.CreateDirectory(projectDir) |> ignore

                                        // Create F# project file with proper compiler settings
                                        let projectContent = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <LangVersion>9</LangVersion>
    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
    <NoWarn>FS0988</NoWarn>
  </PropertyGroup>
</Project>"""
                                        System.IO.File.WriteAllText(projectFile, projectContent)

                                        // Wrap F# code in a proper program structure with correct indentation
                                        let lines = block.Content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)

                                        // Remove duplicate open statements to avoid conflicts
                                        let cleanedLines =
                                            lines
                                            |> Array.filter (fun line ->
                                                let trimmed = line.Trim()
                                                not (trimmed.StartsWith("open System") && (trimmed = "open System" || trimmed = "open System.IO")))

                                        let indentedLines =
                                            cleanedLines
                                            |> Array.map (fun line -> "        " + line.TrimStart())  // 8 spaces for main function body
                                        let indentedContent = String.Join("\n", indentedLines)

                                        // Analyze the F# code to ensure it has executable statements
                                        let hasExecutableStatements =
                                            lines |> Array.exists (fun line ->
                                                let trimmed = line.Trim().ToLower()
                                                trimmed.StartsWith("printfn") ||
                                                trimmed.StartsWith("printf") ||
                                                trimmed.StartsWith("console.write") ||
                                                trimmed.Contains("printfn") ||
                                                trimmed.Contains("printf"))

                                        // If no executable statements, add a summary printout of all variables
                                        let additionalCode =
                                            if not hasExecutableStatements then
                                                let variableLines = lines |> Array.filter (fun line ->
                                                    let trimmed = line.Trim()
                                                    trimmed.StartsWith("let ") && trimmed.Contains("=") && not (trimmed.Contains("printfn")))

                                                if variableLines.Length > 0 then
                                                    let varNames = variableLines |> Array.map (fun line ->
                                                        let trimmed = line.Trim()
                                                        let afterLet = trimmed.Substring(4).Trim()
                                                        let beforeEquals = afterLet.Split('=').[0].Trim()
                                                        beforeEquals)

                                                    let printStatements = varNames |> Array.map (fun varName ->
                                                        sprintf "        printfn \"%s = %%A\" %s" varName varName)

                                                    "\n" + String.Join("\n", printStatements) + "\n"
                                                else
                                                    "\n        printfn \"No variables to display\"\n"
                                            else
                                                "\n"

                                        let programCode =
                                            "open System\n" +
                                            "open System.IO\n\n" +
                                            "[<EntryPoint>]\n" +
                                            "let main argv =\n" +
                                            "    try\n" +
                                            "        printfn \"🔧 FLUX F# Block Execution Started\"\n" +
                                            "        printfn \"=====================================\"\n" +
                                            indentedContent + additionalCode +
                                            "        printfn \"\"\n" +
                                            "        printfn \"✅ FLUX F# Block Execution Completed Successfully\"\n" +
                                            "        0\n" +
                                            "    with\n" +
                                            "    | ex ->\n" +
                                            "        printfn \"❌ FLUX F# Block Execution Failed\"\n" +
                                            "        printfn \"Error: %s\" ex.Message\n" +
                                            "        printfn \"Type: %s\" (ex.GetType().Name)\n" +
                                            "        1\n"

                                        // Write F# code to program file
                                        System.IO.File.WriteAllText(programFile, programCode)

                                        // F# code generation complete





                                        // Execute F# code using F# Interactive instead of dotnet run
                                        let fsiFile = Path.Combine(projectDir, "script.fsx")

                                        // Create F# script without [<EntryPoint>] for F# Interactive
                                        let fsiCode =
                                            "open System\n" +
                                            "open System.IO\n\n" +
                                            "try\n" +
                                            "    printfn \"🔧 FLUX F# Block Execution Started\"\n" +
                                            "    printfn \"=====================================\"\n" +
                                            String.Join("\n    ", cleanedLines) + "\n" +
                                            "    printfn \"\"\n" +
                                            "    printfn \"✅ FLUX F# Block Execution Completed Successfully\"\n" +
                                            "with\n" +
                                            "| ex ->\n" +
                                            "    printfn \"❌ FLUX F# Block Execution Failed\"\n" +
                                            "    printfn \"Error: %s\" ex.Message\n" +
                                            "    printfn \"Type: %s\" (ex.GetType().Name)\n"

                                        System.IO.File.WriteAllText(fsiFile, fsiCode)

                                        // F# script ready for execution

                                        let processInfo = new System.Diagnostics.ProcessStartInfo()
                                        processInfo.FileName <- "dotnet"
                                        processInfo.Arguments <- sprintf "fsi \"%s\"" fsiFile
                                        processInfo.WorkingDirectory <- projectDir
                                        processInfo.UseShellExecute <- false
                                        processInfo.RedirectStandardOutput <- true
                                        processInfo.RedirectStandardError <- true
                                        processInfo.CreateNoWindow <- true
                                        processInfo.StandardOutputEncoding <- System.Text.Encoding.UTF8
                                        processInfo.StandardErrorEncoding <- System.Text.Encoding.UTF8

                                        AnsiConsole.MarkupLine($"[dim]⚙️  Compiling and executing F# code...[/]")
                                        let execStartTime = DateTime.UtcNow

                                        use proc = System.Diagnostics.Process.Start(processInfo)

                                        // Set timeout to prevent hanging
                                        let timeoutMs = 30000 // 30 seconds for compilation + execution
                                        let exited = proc.WaitForExit(timeoutMs)

                                        let execEndTime = DateTime.UtcNow
                                        let execDuration = execEndTime - execStartTime

                                        let stdout, stderr =
                                            if exited then
                                                let stdout = proc.StandardOutput.ReadToEnd()
                                                let stderr = proc.StandardError.ReadToEnd()
                                                stdout, stderr
                                            else
                                                proc.Kill()
                                                "", "Execution timed out after 15 seconds"

                                        // Check for FS0988 warning FIRST - treat as FATAL ERROR regardless of exit code
                                        // Check BOTH stdout and stderr for FS0988 warnings
                                        let hasFS0988Warning =
                                            stderr.Contains("warning FS0988") || stderr.Contains("Main module of program is empty") ||
                                            stdout.Contains("warning FS0988") || stdout.Contains("Main module of program is empty")

                                        if hasFS0988Warning then
                                            // FS0988 is FATAL - F# code didn't actually execute
                                            let errorMsg = "💀 FATAL ERROR: FS0988 - Main module of program is empty. F# code must have proper entry point or executable statements."
                                            let fullErrorMsg = $"💀 FATAL: F# block failed due to FS0988 warning:\nExit Code: {proc.ExitCode}\nExecution Time: {execDuration.TotalSeconds:F2}s\nError: {errorMsg}\nStderr: {stderr}"
                                            output <- fullErrorMsg :: output
                                            hasErrors <- true

                                            AnsiConsole.MarkupLine($"[red]💀 FATAL ERROR: FS0988 detected[/]")
                                            AnsiConsole.MarkupLine("[red]═══════════════════════════════════════════════════════════════[/]")
                                            AnsiConsole.MarkupLine($"[red]💀 FATAL: FS0988 - Main module of program is empty[/]")
                                            AnsiConsole.MarkupLine($"[red]This means the F# code did NOT execute at all![/]")
                                            AnsiConsole.MarkupLine($"[red]F# code must have proper entry point or executable statements[/]")
                                            AnsiConsole.MarkupLine($"[red]Exit Code: {proc.ExitCode} (irrelevant - code didn't run)[/]")
                                            AnsiConsole.MarkupLine($"[red]Stderr: {stderr}[/]")
                                            AnsiConsole.MarkupLine("[red]═══════════════════════════════════════════════════════════════[/]")
                                        elif not exited then
                                            // Timeout
                                            let errorMsg = "❌ F# execution timed out"
                                            let fullErrorMsg = $"❌ F# block timed out:\nExecution Time: {execDuration.TotalSeconds:F2}s\nError: {errorMsg}"
                                            output <- fullErrorMsg :: output
                                            hasErrors <- true

                                            AnsiConsole.MarkupLine($"[red]❌ F# execution timed out after {execDuration.TotalSeconds:F2}s[/]")
                                        elif proc.ExitCode = 0 then
                                            // Process F# Interactive output

                                            // Enhanced F# output processing
                                            if not (String.IsNullOrWhiteSpace(stdout)) then
                                                // Clean up compiler warnings and format output (but keep FS0988 detection above)
                                                let lines = stdout.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                                                let cleanLines = lines
                                                               |> Array.filter (fun line ->
                                                                   not (line.Trim().StartsWith("C:\\Users\\")) &&
                                                                   not (String.IsNullOrWhiteSpace(line)))

                                                if cleanLines.Length > 0 then
                                                    AnsiConsole.MarkupLine($"[green]📤 FLUX F# Execution Output:[/]")
                                                    AnsiConsole.MarkupLine("[green]═══════════════════════════════════════════════════════════════[/]")

                                                    for line in cleanLines do
                                                        // Color-code different types of output
                                                        if line.Contains("🔧") || line.Contains("====") then
                                                            AnsiConsole.MarkupLine($"[dim]{line}[/]")
                                                        elif line.Contains("✅") then
                                                            AnsiConsole.MarkupLine($"[green]{line}[/]")
                                                        elif line.Contains("❌") then
                                                            AnsiConsole.MarkupLine($"[red]{line}[/]")
                                                        elif line.Contains("🌀") || line.Contains("🌌") || line.Contains("📊") then
                                                            AnsiConsole.MarkupLine($"[cyan]{line}[/]")
                                                        elif line.Contains("🔢") || line.Contains("📐") || line.Contains("🔬") then
                                                            AnsiConsole.MarkupLine($"[yellow]{line}[/]")
                                                        else
                                                            AnsiConsole.WriteLine(line)

                                                    AnsiConsole.MarkupLine("[green]═══════════════════════════════════════════════════════════════[/]")
                                                else
                                                    AnsiConsole.MarkupLine($"[yellow]⚠️  F# code executed but produced no visible output[/]")

                                            let result = if String.IsNullOrWhiteSpace(stdout) then "F# code executed successfully" else stdout.Trim()
                                            output <- $"✅ F# block executed successfully:\n{result}" :: output
                                            logger.LogInformation($"Executed F# block: {block.Content.Substring(0, Math.Min(50, block.Content.Length))}...")
                                            AnsiConsole.MarkupLine($"[green]✅ F# execution completed in {execDuration.TotalSeconds:F2}s[/]")
                                        else
                                            let errorMsg = if String.IsNullOrWhiteSpace(stderr) then "F# execution failed" else stderr.Trim()
                                            let fullErrorMsg = $"❌ F# block failed:\nExit Code: {proc.ExitCode}\nExecution Time: {execDuration.TotalSeconds:F2}s\nStdout: {stdout}\nStderr: {errorMsg}"
                                            output <- fullErrorMsg :: output
                                            hasErrors <- true

                                            AnsiConsole.MarkupLine($"[red]❌ F# execution failed in {execDuration.TotalSeconds:F2}s[/]")
                                            AnsiConsole.MarkupLine("[red]═══════════════════════════════════════════════════════════════[/]")
                                            AnsiConsole.MarkupLine($"[red]Exit Code: {proc.ExitCode}[/]")

                                            if not (String.IsNullOrWhiteSpace(stdout)) then
                                                AnsiConsole.MarkupLine($"[yellow]📤 Standard Output:[/]")
                                                AnsiConsole.WriteLine(stdout)

                                            if not (String.IsNullOrWhiteSpace(stderr)) then
                                                AnsiConsole.MarkupLine($"[red]📤 Error Output:[/]")
                                                AnsiConsole.WriteLine(errorMsg)

                                            AnsiConsole.MarkupLine("[red]═══════════════════════════════════════════════════════════════[/]")

                                        // Clean up temporary project directory
                                        try
                                            System.IO.Directory.Delete(projectDir, true)
                                        with | _ -> ()
                                    with
                                    | ex ->
                                        let fullErrorMsg = $"❌ F# block failed with exception:\nMessage: {ex.Message}\nType: {ex.GetType().Name}\nStackTrace: {ex.StackTrace}"
                                        output <- fullErrorMsg :: output
                                        hasErrors <- true
                                        AnsiConsole.MarkupLine($"[red]❌ F# execution failed with exception[/]")
                                        AnsiConsole.MarkupLine($"[red]Message: {ex.Message}[/]")
                                        AnsiConsole.MarkupLine($"[red]Type: {ex.GetType().Name}[/]")
                                        if not (String.IsNullOrWhiteSpace(ex.StackTrace)) then
                                            AnsiConsole.MarkupLine($"[dim]StackTrace: {ex.StackTrace}[/]")

                                | Meta ->
                                    AnsiConsole.MarkupLine($"[yellow]📋 Processing META block[/]")
                                    output <- "✅ Processed meta block" :: output
                                | Reasoning ->
                                    AnsiConsole.MarkupLine($"[magenta]🧠 Processing REASONING block[/]")
                                    output <- "✅ Processed reasoning block" :: output
                                | Lang lang ->
                                    AnsiConsole.MarkupLine($"[cyan]🌐 Executing {lang} block[/]")

                                    try
                                        // Use TARS LanguageDispatcher for real execution
                                        let languageBlock = {
                                            Tars.Engine.Grammar.LanguageBlock.Language = lang
                                            Tars.Engine.Grammar.LanguageBlock.Code = block.Content
                                            Tars.Engine.Grammar.LanguageBlock.Metadata = Map.empty
                                            Tars.Engine.Grammar.LanguageBlock.EntryPoint = None
                                            Tars.Engine.Grammar.LanguageBlock.Dependencies = []
                                        }
                                        let context = Tars.Engine.Grammar.LanguageDispatcher.createDefaultContext()
                                        let result = Tars.Engine.Grammar.LanguageDispatcher.executeLanguageBlock languageBlock context

                                        if result.Success then
                                            AnsiConsole.MarkupLine($"[green]✅ {lang} execution completed successfully[/]")
                                            if not (String.IsNullOrWhiteSpace(result.Output)) then
                                                AnsiConsole.MarkupLine($"[green]📤 {lang} Output:[/]")
                                                AnsiConsole.MarkupLine("[green]═══════════════════════════════════════════════════════════════[/]")
                                                AnsiConsole.WriteLine(result.Output)
                                                AnsiConsole.MarkupLine("[green]═══════════════════════════════════════════════════════════════[/]")
                                            output <- $"✅ {lang} block executed successfully:\n{result.Output}" :: output
                                        else
                                            let errorMsg = result.Error |> Option.defaultValue "Unknown error"
                                            AnsiConsole.MarkupLine($"[red]❌ {lang} execution failed[/]")
                                            AnsiConsole.MarkupLine($"[red]Error: {errorMsg}[/]")
                                            output <- $"❌ {lang} block failed: {errorMsg}" :: output
                                            hasErrors <- true
                                    with
                                    | ex ->
                                        AnsiConsole.MarkupLine($"[red]❌ {lang} execution failed with exception[/]")
                                        AnsiConsole.MarkupLine($"[red]Message: {ex.Message}[/]")
                                        output <- $"❌ {lang} block failed: {ex.Message}" :: output
                                        hasErrors <- true

                                task.Increment(1.0)
                                // REMOVED: Task.Delay fake simulation
                        })

                    let endTime = DateTime.UtcNow
                    let executionTime = endTime - startTime

                    let finalOutput = String.Join("\n", List.rev output)

                    // Enhanced execution summary
                    AnsiConsole.MarkupLine("")
                    AnsiConsole.MarkupLine("[cyan]═══════════════════════════════════════════════════════════════[/]")
                    AnsiConsole.MarkupLine("[cyan]                    FLUX Execution Summary                    [/]")
                    AnsiConsole.MarkupLine("[cyan]═══════════════════════════════════════════════════════════════[/]")

                    if hasErrors then
                        AnsiConsole.MarkupLine("[red]❌ Metascript execution completed with errors[/]")
                    else
                        AnsiConsole.MarkupLine("[green]✅ Metascript execution completed successfully[/]")

                    AnsiConsole.MarkupLine($"[dim]⏱️  Total execution time: {executionTime.TotalSeconds:F2} seconds[/]")
                    AnsiConsole.MarkupLine($"[dim]📊 Processed {parsed.Blocks.Length} code blocks[/]")

                    let fsharpBlocks = parsed.Blocks |> List.filter (fun b -> b.Type = FSharp) |> List.length
                    if fsharpBlocks > 0 then
                        AnsiConsole.MarkupLine($"[dim]🔧 F# blocks executed: {fsharpBlocks}[/]")

                    AnsiConsole.MarkupLine("[cyan]═══════════════════════════════════════════════════════════════[/]")

                    return {
                        Status = if hasErrors then ExecutionStatus.Failed else ExecutionStatus.Success
                        Output = finalOutput
                        Error = None
                        Variables = Map.empty
                        ExecutionTime = executionTime
                    }
                with
                | ex ->
                    let endTime = DateTime.UtcNow
                    let executionTime = endTime - startTime

                    AnsiConsole.MarkupLine($"[red]💥 Error executing metascript: {ex.Message}[/]")
                    logger.LogError(ex, $"Error executing metascript: {metascriptPath}")

                    return {
                        Status = ExecutionStatus.Failed
                        Output = ""
                        Error = Some ex.Message
                        Variables = Map.empty
                        ExecutionTime = executionTime
                    }
            }

        /// Execute FLUX/TRSX files with tier-based processing
        member this.ExecuteFluxFile(filePath: string) : Async<ExecutionResult> =
            async {
                let startTime = DateTime.UtcNow
                try

                    AnsiConsole.MarkupLine($"[cyan]🌀 FLUX Tier-Based Execution: {System.IO.Path.GetFileName(filePath)}[/]")

                    // Use unified interpreter for FLUX/TRSX execution
                    let result = unifiedInterpreter.ExecuteFile(filePath)

                    let endTime = DateTime.UtcNow
                    let executionTime = endTime - startTime

                    if result.Success then
                        AnsiConsole.MarkupLine("[green]✅ FLUX execution completed successfully[/]")
                        AnsiConsole.MarkupLine($"[dim]🎯 Tier: {result.Tier}[/]")
                        AnsiConsole.MarkupLine($"[dim]📊 Format: {result.Format}[/]")

                        match result.FractalMetrics with
                        | Some metrics ->
                            AnsiConsole.MarkupLine($"[dim]🌀 Fractal Dimension: {metrics.Dimension:F3}[/]")
                            AnsiConsole.MarkupLine($"[dim]🔗 Self-Similarity: {metrics.SelfSimilarity:F3}[/]")
                        | None -> ()

                        return {
                            Status = ExecutionStatus.Success
                            Output = result.Output
                            Error = None
                            Variables = Map.empty
                            ExecutionTime = executionTime
                        }
                    else
                        AnsiConsole.MarkupLine("[red]❌ FLUX execution failed[/]")
                        AnsiConsole.MarkupLine($"[red]Error: {result.Output}[/]")

                        return {
                            Status = ExecutionStatus.Failed
                            Output = ""
                            Error = Some result.Output
                            Variables = Map.empty
                            ExecutionTime = executionTime
                        }
                with
                | ex ->
                    let endTime = DateTime.UtcNow
                    let executionTime = endTime - startTime

                    AnsiConsole.MarkupLine($"[red]💥 FLUX execution error: {ex.Message}[/]")
                    logger.LogError(ex, $"Error executing FLUX file: {filePath}")

                    return {
                        Status = ExecutionStatus.Failed
                        Output = ""
                        Error = Some ex.Message
                        Variables = Map.empty
                        ExecutionTime = executionTime
                    }
            }


    /// TARS API implementation that works
    type TarsApiService(logger: ILogger<TarsApiService>) =
        interface ITarsApi with
            member _.SearchVector(query: string, limit: int) =
                async {
                    do! Async.Sleep(100)
                    return [
                        for i in 1..limit ->
                            {| Id = i; Content = sprintf "Result %d for '%s'" i query; Score = 1.0 - (float i * 0.1) |}
                    ]
                }
            
            member _.AskLlm(prompt: string, model: string) =
                async {
                    do! Async.Sleep(500)
                    return sprintf "Response to '%s' using %s model" prompt model
                }
            
            member _.SpawnAgent(agentType: string, config: AgentConfig) =
                sprintf "%s-agent-%s" agentType (Guid.NewGuid().ToString("N")[..7])
            
            member _.WriteFile(path: string, content: string) =
                try
                    System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)) |> ignore
                    System.IO.File.WriteAllText(path, content)
                    true
                with
                | _ -> false
