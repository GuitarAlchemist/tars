namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Handler for command blocks.
/// </summary>
type CommandBlockHandler(logger: ILogger<CommandBlockHandler>) =
    inherit BlockHandlerBase(logger, MetascriptBlockType.Command, 90)
    
    /// <summary>
    /// Executes a command.
    /// </summary>
    /// <param name="command">The command to execute.</param>
    /// <param name="workingDirectory">The working directory.</param>
    /// <returns>The output, error, and success status.</returns>
    let executeCommand (command: string) (workingDirectory: string) =
        try
            let processStartInfo = ProcessStartInfo()
            processStartInfo.FileName <- "cmd.exe"
            processStartInfo.Arguments <- $"/c {command}"
            processStartInfo.WorkingDirectory <- workingDirectory
            processStartInfo.RedirectStandardOutput <- true
            processStartInfo.RedirectStandardError <- true
            processStartInfo.UseShellExecute <- false
            processStartInfo.CreateNoWindow <- true
            
            let proc = Process.Start(processStartInfo)
            let output = proc.StandardOutput.ReadToEnd()
            let error = proc.StandardError.ReadToEnd()
            proc.WaitForExit()
            
            (output, error, proc.ExitCode = 0)
        with
        | ex ->
            ("", ex.ToString(), false)
    
    /// <summary>
    /// Executes a command block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    override this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            try
                // Get the command
                let command = block.Content.Trim()
                
                // Execute the command
                let (output, error, success) = executeCommand command context.WorkingDirectory
                
                // Check for errors
                if not success then
                    return this.CreateFailureResult(block, error, output)
                else
                    return this.CreateSuccessResult(block, output)
            with
            | ex ->
                logger.LogError(ex, "Error executing command block")
                return this.CreateFailureResult(block, ex.ToString())
        }
