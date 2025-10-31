namespace TarsEngine.FSharp.Metascripts.Services

open System
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascripts.Core
open TarsEngine.FSharp.Metascripts.Discovery

// Execution context for metascript processing
type MetascriptExecutionContext = {
    MetascriptName: string
    SessionId: string
    StartTime: DateTime
    Logger: ILogger
    Variables: Map<string, obj>
    OutputPath: string option
}

// Execution result
type MetascriptExecutionResult = {
    Success: bool
    Output: string
    Error: string option
    Variables: Map<string, obj>
}

/// <summary>
/// Implementation of metascript services.
/// </summary>
type MetascriptService(
    registry: MetascriptRegistry, 
    manager: MetascriptManager, 
    discovery: MetascriptDiscovery,
    logger: ILogger<MetascriptService>) =
    
    interface IMetascriptService with
        member _.DiscoverMetascriptsAsync(directory: string) =
            task {
                try
                    logger.LogInformation $"Starting metascript discovery in: %s{directory}"
                    let! result = discovery.DiscoverMetascriptsAsync(directory, true)
                    match result with
                    | Ok metascripts ->
                        logger.LogInformation $"Discovery completed. Found %d{metascripts.Length} metascripts"
                        return Ok metascripts
                    | Error error ->
                        logger.LogError $"Discovery failed: %s{error}"
                        return Error error
                with
                | ex ->
                    let error = $"Error during discovery: %s{ex.Message}"
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.ListMetascriptsAsync() =
            task {
                try
                    logger.LogInformation("Listing all registered metascripts")
                    let metascripts = registry.GetAllMetascripts()
                    logger.LogInformation $"Found %d{metascripts.Length} registered metascripts"
                    return Ok metascripts
                with
                | ex ->
                    let error = $"Error listing metascripts: %s{ex.Message}"
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.GetMetascriptAsync(name: string) =
            task {
                try
                    logger.LogDebug $"Getting metascript: %s{name}"
                    let metascript = registry.GetMetascript(name)
                    return Ok metascript
                with
                | ex ->
                    let error = $"Error getting metascript %s{name}: %s{ex.Message}"
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member this.ExecuteMetascriptAsync(name: string) =
            task {
                try
                    logger.LogInformation $"🚀 TARS: Starting metascript execution: %s{name}"

                    match registry.GetMetascript(name) with
                    | Some registered ->
                        let startTime = DateTime.UtcNow
                        let sessionId = Guid.NewGuid().ToString("N")[..7]

                        // Update usage statistics
                        registry.UpdateUsage(name) |> ignore

                        // Create execution context with logging
                        let executionContext = {
                            MetascriptName = name
                            SessionId = sessionId
                            StartTime = startTime
                            Logger = logger
                            Variables = Map.empty
                            OutputPath = None
                        }

                        // Execute the metascript with real processing
                        let! executionResult = this.executeMetascriptContent registered.Source executionContext

                        let endTime = DateTime.UtcNow
                        let executionTime = endTime - startTime

                        let result = {
                            Id = sessionId
                            MetascriptId = registered.Source.Id
                            Status = if executionResult.Success then Completed else Failed
                            Output = executionResult.Output
                            Error = executionResult.Error
                            Variables = executionResult.Variables
                            ExecutionTime = executionTime
                            StartTime = startTime
                            EndTime = Some endTime
                        }

                        logger.LogInformation $"✅ TARS: Metascript execution completed: %s{name} (took %d{int executionTime.TotalMilliseconds}ms)"

                        return Ok result
                    | None ->
                        let error = $"Metascript not found: %s{name}"
                        logger.LogWarning(error)
                        return Error error
                with
                | ex ->
                    let error = $"Error executing metascript %s{name}: %s{ex.Message}"
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.GetStatisticsAsync() =
            task {
                try
                    logger.LogDebug("Getting metascript statistics")
                    let stats = registry.GetStatistics()
                    return Ok stats
                with
                | ex ->
                    let error = $"Error getting statistics: %s{ex.Message}"
                    logger.LogError(ex, error)
                    return Error error
            }
        
        member _.ValidateMetascriptAsync(source: MetascriptSource) =
            task {
                try
                    logger.LogDebug $"Validating metascript: %s{source.Name}"
                    let validatedSource = manager.ValidateMetascript(source)
                    return Ok validatedSource
                with
                | ex ->
                    let error = $"Error validating metascript %s{source.Name}: %s{ex.Message}"
                    logger.LogError(ex, error)
                    return Error error
            }

        member _.RegisterMetascriptAsync(source: MetascriptSource) =
            task {
                try
                    logger.LogInformation $"Registering metascript: %s{source.Name}"

                    // Register with the registry (it creates the RegisteredMetascript internally)
                    let registered = registry.RegisterMetascript(source)

                    logger.LogInformation $"Metascript registered successfully: %s{source.Name}"
                    return Ok ()
                with
                | ex ->
                    let error = $"Error registering metascript %s{source.Name}: %s{ex.Message}"
                    logger.LogError(ex, error)
                    return Error error
            }

    // Real metascript execution implementation
    member private this.executeMetascriptContent (source: MetascriptSource) (context: MetascriptExecutionContext) =
        task {
            try
                context.Logger.LogInformation(sprintf "[%s] 🚀 SYSTEM_START | Metascript Execution | Starting TARS metascript: %s"
                    (DateTime.Now.ToString("HH:mm:ss.fff")) context.MetascriptName)

                // Create detailed execution log
                let logBuilder = System.Text.StringBuilder()
                let appendLog (message: string) =
                    let timestamp = DateTime.Now.ToString("HH:mm:ss.fff")
                    let logLine = $"[%s{timestamp}] %s{message}"
                    logBuilder.AppendLine(logLine) |> ignore
                    context.Logger.LogInformation(logLine)

                appendLog "=================================================================================="
                appendLog "TARS METASCRIPT EXECUTION LOG"
                appendLog "=================================================================================="
                appendLog (sprintf "Start Time: %s" (context.StartTime.ToString("yyyy-MM-dd HH:mm:ss")))
                appendLog $"Metascript: %s{context.MetascriptName}"
                appendLog $"Session ID: %s{context.SessionId}"
                appendLog ""

                // Parse metascript content
                appendLog "📋 PHASE_START | METASCRIPT_PARSING | Parsing metascript content"
                let parsedSections = this.parseMetascriptSections source.Content
                appendLog $"📊 PARSING_RESULT | Found %d{parsedSections.Length} sections to execute"

                let mutable variables = context.Variables
                let mutable executionOutput = System.Text.StringBuilder()

                // Execute each section with early termination on error
                let rec executeSections (sections: {| SectionType: string; Content: string |} list) (sectionIndex: int) (currentVariables: Map<string, obj>) =
                    task {
                        match sections with
                        | [] ->
                            return Ok currentVariables
                        | section :: remainingSections ->
                            appendLog $"🔧 SECTION_START | Section %d{sectionIndex} | Type: %s{section.SectionType}"

                            match section.SectionType with
                            | "fsharp" ->
                                appendLog "💻 F#_EXECUTION | Executing F# code block"
                                let! fsharpResult = this.executeFSharpCode section.Content currentVariables
                                if fsharpResult.Success then
                                    appendLog (sprintf "✅ F#_SUCCESS | F# code executed successfully")
                                    executionOutput.AppendLine(fsharpResult.Output) |> ignore
                                    appendLog $"✅ SECTION_END | Section %d{sectionIndex} completed"
                                    return! executeSections remainingSections (sectionIndex + 1) fsharpResult.Variables
                                else
                                    appendLog (sprintf "❌ F#_ERROR | F# execution failed: %s" (fsharpResult.Error |> Option.defaultValue "Unknown error"))
                                    return Error fsharpResult.Error

                            | "yaml" ->
                                appendLog "⚙️ YAML_PROCESSING | Processing YAML configuration"
                                let yamlResult = this.processYamlSection section.Content
                                appendLog "✅ YAML_SUCCESS | YAML configuration processed"
                                let updatedVariables = Map.fold (fun acc key value -> Map.add key value acc) currentVariables yamlResult
                                appendLog $"✅ SECTION_END | Section %d{sectionIndex} completed"
                                return! executeSections remainingSections (sectionIndex + 1) updatedVariables

                            | "command" ->
                                appendLog "🛠 COMMAND_EXECUTION | Executing shell command block"
                                let! commandResult = this.executeCommand section.Content currentVariables
                                if commandResult.Success then
                                    if not (String.IsNullOrWhiteSpace(commandResult.Output)) then
                                        appendLog $"📤 COMMAND_OUTPUT | %s{commandResult.Output.Trim()}"
                                    appendLog (sprintf "✅ COMMAND_SUCCESS | Command executed successfully")
                                    appendLog $"✅ SECTION_END | Section %d{sectionIndex} completed"
                                    return! executeSections remainingSections (sectionIndex + 1) commandResult.Variables
                                else
                                    appendLog (sprintf "❌ COMMAND_ERROR | %s" (commandResult.Error |> Option.defaultValue "Unknown error"))
                                    return Error commandResult.Error

                            | "markdown" ->
                                appendLog "📝 MARKDOWN_PROCESSING | Processing markdown documentation"
                                executionOutput.AppendLine(section.Content) |> ignore
                                appendLog "✅ MARKDOWN_SUCCESS | Markdown content processed"
                                appendLog $"✅ SECTION_END | Section %d{sectionIndex} completed"
                                return! executeSections remainingSections (sectionIndex + 1) currentVariables

                            | _ ->
                                appendLog $"⚠️ UNKNOWN_SECTION | Unknown section type: %s{section.SectionType}"
                                appendLog $"✅ SECTION_END | Section %d{sectionIndex} completed"
                                return! executeSections remainingSections (sectionIndex + 1) currentVariables
                    }

                let! executionResult = executeSections parsedSections 1 variables

                match executionResult with
                | Ok finalVariables ->
                    variables <- finalVariables

                    let endTime = DateTime.Now
                    let duration = endTime - context.StartTime

                    appendLog ""
                    appendLog "=================================================================================="
                    appendLog "METASCRIPT EXECUTION SUMMARY"
                    appendLog "=================================================================================="
                    appendLog (sprintf "End Time: %s" (endTime.ToString("yyyy-MM-dd HH:mm:ss")))
                    appendLog $"Total Duration: %.2f{duration.TotalSeconds} seconds"
                    appendLog $"Sections Executed: %d{parsedSections.Length}"
                    appendLog $"Variables Created: %d{variables.Count}"
                    appendLog (sprintf "Success Rate: 100%%")
                    appendLog ""
                    appendLog "✅ SYSTEM_END | Metascript execution completed successfully"
                    appendLog "=================================================================================="

                    // Save execution log if output path is specified
                    match context.OutputPath with
                    | Some outputPath ->
                        let logPath = Path.Combine(outputPath, "tars.log")
                        Directory.CreateDirectory(outputPath) |> ignore
                        File.WriteAllText(logPath, logBuilder.ToString())
                        appendLog $"📄 LOG_SAVED | Execution log saved: %s{logPath}"
                    | None -> ()

                    return {
                        Success = true
                        Output = logBuilder.ToString()
                        Error = None
                        Variables = variables
                    }

                | Error errorMsg ->
                    return {
                        Success = false
                        Output = logBuilder.ToString()
                        Error = errorMsg
                        Variables = variables
                    }

            with
            | ex ->
                context.Logger.LogError(ex, $"❌ EXECUTION_ERROR | Metascript execution failed: %s{ex.Message}")
                return {
                    Success = false
                    Output = $"Metascript execution failed: %s{ex.Message}"
                    Error = Some ex.Message
                    Variables = context.Variables
                }
        }

    // Parse metascript sections
    member private this.parseMetascriptSections (content: string) : {| SectionType: string; Content: string |} list =
        let sections = System.Collections.Generic.List<{| SectionType: string; Content: string |}>()

        // Split content by code blocks
        let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        let mutable currentSection = ""
        let mutable currentContent = System.Text.StringBuilder()
        let mutable inCodeBlock = false

        for line in lines do
            if line.StartsWith("```") then
                if inCodeBlock then
                    // End of code block
                    if currentSection <> "" then
                        sections.Add({| SectionType = currentSection; Content = currentContent.ToString().Trim() |})
                    currentSection <- ""
                    currentContent.Clear() |> ignore
                    inCodeBlock <- false
                else
                    // Start of code block
                    let sectionType = line.Substring(3).Trim().ToLower()
                    currentSection <- if sectionType = "" then "text" else sectionType
                    inCodeBlock <- true
            elif inCodeBlock then
                currentContent.AppendLine(line) |> ignore
            else
                // Regular markdown content
                if currentSection <> "markdown" then
                    if currentSection <> "" then
                        sections.Add({| SectionType = currentSection; Content = currentContent.ToString().Trim() |})
                    currentSection <- "markdown"
                    currentContent.Clear() |> ignore
                currentContent.AppendLine(line) |> ignore

        // Add final section
        if currentSection <> "" then
            sections.Add({| SectionType = currentSection; Content = currentContent.ToString().Trim() |})

        sections |> Seq.toList

    // Execute F# code blocks using dotnet fsi to ensure real evaluation.
    member private this.executeFSharpCode (code: string) (variables: Map<string, obj>) =
        task {
            try
                // REAL F# EXECUTION using F# Interactive
                let tempFile = Path.GetTempFileName() + ".fsx"

                // Prepare variable setup
                let variableSetup =
                    variables
                    |> Map.toSeq
                    |> Seq.map (fun (k, v) -> $"let %s{k} = %A{v}")
                    |> String.concat "\n"

                let fullCode = if String.IsNullOrEmpty(variableSetup) then code else variableSetup + "\n\n" + code
                File.WriteAllText(tempFile, fullCode)

                // Execute using dotnet fsi (REAL execution)
                let psi = System.Diagnostics.ProcessStartInfo()
                psi.FileName <- "dotnet"
                psi.Arguments <- $"fsi \"%s{tempFile}\""
                psi.UseShellExecute <- false
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.CreateNoWindow <- true

                use proc = System.Diagnostics.Process.Start(psi)
                proc.WaitForExit(30000) |> ignore // 30 second timeout

                let output = proc.StandardOutput.ReadToEnd()
                let error = proc.StandardError.ReadToEnd()

                // Clean up temp file
                try File.Delete(tempFile) with | _ -> ()

                if proc.ExitCode = 0 then
                    return {
                        Success = true
                        Output = output
                        Error = None
                        Variables = variables
                    }
                else
                    return {
                        Success = false
                        Output = output
                        Error = Some error
                        Variables = variables
                    }
            with
            | ex ->
                return {
                    Success = false
                    Output = ""
                    Error = Some ex.Message
                    Variables = variables
                }
        }

    // Execute command sections via the platform shell.
    member private this.executeCommand (commandContent: string) (variables: Map<string, obj>) =
        task {
            try
                let scriptPath =
                    if OperatingSystem.IsWindows() then
                        let path = Path.ChangeExtension(Path.GetTempFileName(), ".cmd")
                        File.WriteAllText(path, commandContent)
                        path
                    else
                        let path = Path.ChangeExtension(Path.GetTempFileName(), ".sh")
                        let script = "#!/bin/bash\nset -euo pipefail\n" + commandContent
                        File.WriteAllText(path, script)
                        File.SetAttributes(path, File.GetAttributes(path) ||| FileAttributes.Normal)
                        path

                let psi = System.Diagnostics.ProcessStartInfo()
                if OperatingSystem.IsWindows() then
                    psi.FileName <- "cmd.exe"
                    psi.Arguments <- $"/c \"%s{scriptPath}\""
                else
                    psi.FileName <- "/bin/bash"
                    psi.Arguments <- $"\"%s{scriptPath}\""

                psi.UseShellExecute <- false
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.CreateNoWindow <- true

                use proc = System.Diagnostics.Process.Start(psi)
                proc.WaitForExit(30000) |> ignore

                let output = proc.StandardOutput.ReadToEnd()
                let error = proc.StandardError.ReadToEnd()

                try File.Delete(scriptPath) with | _ -> ()

                if proc.ExitCode = 0 then
                    return {
                        Success = true
                        Output = output
                        Error = None
                        Variables = variables
                    }
                else
                    return {
                        Success = false
                        Output = output
                        Error = Some error
                        Variables = variables
                    }
            with
            | ex ->
                return {
                    Success = false
                    Output = ""
                    Error = Some ex.Message
                    Variables = variables
                }
        }

    // Process YAML section
    member private this.processYamlSection (yamlContent: string) =
        // Simple YAML processing - in a full implementation would use a YAML parser
        let variables = System.Collections.Generic.Dictionary<string, obj>()

        let lines = yamlContent.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        for line in lines do
            let trimmed = line.Trim()
            if trimmed.Contains(":") && not (trimmed.StartsWith("#")) then
                let parts = trimmed.Split([|':'|], 2)
                if parts.Length = 2 then
                    let key = parts.[0].Trim()
                    let value = parts.[1].Trim().Trim('"')
                    variables.[key] <- box value

        variables |> Seq.map (|KeyValue|) |> Map.ofSeq
