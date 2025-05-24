namespace TarsEngine.FSharp.Core.Simple.Metascript

open System
open System.Diagnostics
open System.IO
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Simple metascript executor.
/// </summary>
type MetascriptExecutor(logger: ILogger<MetascriptExecutor>) =
    
    /// <summary>
    /// Executes a metascript.
    /// </summary>
    member _.ExecuteAsync(metascript: Metascript, context: MetascriptContext) =
        task {
            let stopwatch = Stopwatch.StartNew()
            let output = StringBuilder()
            let mutable variables = context.Variables
            let mutable error = None
            let mutable status = MetascriptExecutionStatus.Success
            
            try
                logger.LogInformation($"Executing metascript: {metascript.Name}")
                
                for block in metascript.Blocks do
                    match block with
                    | ConfigBlock config ->
                        logger.LogDebug("Processing CONFIG block")
                        output.AppendLine("Processing configuration...") |> ignore
                        
                        // Apply configuration to context
                        for kvp in config do
                            let variable = MetascriptVariable.create kvp.Key kvp.Value
                            variables <- variables.Add(kvp.Key, variable)
                    
                    | FSharpBlock code ->
                        logger.LogDebug("Processing FSHARP block")
                        output.AppendLine("Executing F# code...") |> ignore
                        
                        // For now, just simulate F# execution
                        // In a real implementation, this would compile and execute F# code
                        output.AppendLine($"// F# Code:\n{code}") |> ignore
                        output.AppendLine("// F# execution completed (simulated)") |> ignore
                    
                    | CommandBlock command ->
                        logger.LogDebug("Processing COMMAND block")
                        output.AppendLine($"Executing command: {command}") |> ignore
                        
                        // Execute shell command
                        let! commandResult = executeCommand command context.WorkingDirectory
                        output.AppendLine(commandResult) |> ignore
                    
                    | TextBlock text ->
                        logger.LogDebug("Processing TEXT block")
                        output.AppendLine(text) |> ignore
                
                logger.LogInformation($"Metascript execution completed: {metascript.Name}")
                
            with
            | ex ->
                logger.LogError(ex, $"Error executing metascript: {metascript.Name}")
                error <- Some ex.Message
                status <- MetascriptExecutionStatus.Failed
            
            stopwatch.Stop()
            
            return {
                Status = status
                Output = output.ToString()
                Error = error
                Variables = variables
                ExecutionTime = stopwatch.Elapsed
            }
        }
    
    /// <summary>
    /// Creates a new execution context.
    /// </summary>
    member _.CreateContextAsync(workingDirectory: string, ?variables: Map<string, MetascriptVariable>) =
        task {
            let context = MetascriptContext.create workingDirectory
            match variables with
            | Some vars -> return { context with Variables = vars }
            | None -> return context
        }
    
    /// <summary>
    /// Executes a shell command.
    /// </summary>
    member private _.executeCommand(command: string, workingDirectory: string) =
        task {
            try
                let processInfo = ProcessStartInfo()
                processInfo.FileName <- "cmd.exe"
                processInfo.Arguments <- $"/c {command}"
                processInfo.WorkingDirectory <- workingDirectory
                processInfo.UseShellExecute <- false
                processInfo.RedirectStandardOutput <- true
                processInfo.RedirectStandardError <- true
                processInfo.CreateNoWindow <- true
                
                use process = Process.Start(processInfo)
                let! output = process.StandardOutput.ReadToEndAsync()
                let! error = process.StandardError.ReadToEndAsync()
                
                process.WaitForExit()
                
                if process.ExitCode = 0 then
                    return output
                else
                    return $"Command failed with exit code {process.ExitCode}:\n{error}"
            with
            | ex ->
                return $"Error executing command: {ex.Message}"
        }
