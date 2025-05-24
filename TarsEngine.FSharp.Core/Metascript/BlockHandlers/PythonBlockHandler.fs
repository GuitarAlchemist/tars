namespace TarsEngine.FSharp.Core.Metascript.BlockHandlers

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript

/// <summary>
/// Handler for Python blocks.
/// </summary>
type PythonBlockHandler(logger: ILogger<PythonBlockHandler>) =
    inherit BlockHandlerBase(logger, MetascriptBlockType.Python, 80)
    
    /// <summary>
    /// Executes Python code.
    /// </summary>
    /// <param name="code">The code to execute.</param>
    /// <param name="workingDirectory">The working directory.</param>
    /// <returns>The output, error, and success status.</returns>
    let executePythonCode (code: string) (workingDirectory: string) =
        try
            // Create a temporary file for the code
            let tempFile = Path.Combine(Path.GetTempPath(), $"metascript_python_{Guid.NewGuid()}.py")
            File.WriteAllText(tempFile, code)
            
            // Create process info
            let processStartInfo = ProcessStartInfo()
            processStartInfo.FileName <- "python"
            processStartInfo.Arguments <- tempFile
            processStartInfo.WorkingDirectory <- workingDirectory
            processStartInfo.RedirectStandardOutput <- true
            processStartInfo.RedirectStandardError <- true
            processStartInfo.UseShellExecute <- false
            processStartInfo.CreateNoWindow <- true
            
            // Execute the process
            let process = Process.Start(processStartInfo)
            let output = process.StandardOutput.ReadToEnd()
            let error = process.StandardError.ReadToEnd()
            process.WaitForExit()
            
            // Delete the temporary file
            File.Delete(tempFile)
            
            (output, error, process.ExitCode = 0)
        with
        | ex ->
            ("", ex.ToString(), false)
    
    /// <summary>
    /// Executes a Python block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    override this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            try
                // Execute the Python code
                let (output, error, success) = executePythonCode block.Content context.WorkingDirectory
                
                // Check for errors
                if not success then
                    return this.CreateFailureResult(block, error, output)
                else
                    return this.CreateSuccessResult(block, output)
            with
            | ex ->
                logger.LogError(ex, "Error executing Python block")
                return this.CreateFailureResult(block, ex.ToString())
        }
