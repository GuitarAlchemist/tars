namespace TarsEngine.FSharp.Core.Working.Metascript

open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Handler for COMMAND blocks in metascripts.
/// </summary>
type CommandBlockHandler(logger: ILogger<CommandBlockHandler>) =
    inherit BlockHandlerBase(logger, "COMMAND", 80)
    
    override _.ValidateBlock(content: string) =
        if System.String.IsNullOrWhiteSpace(content) then
            Error (createError "COMMAND block cannot be empty" None)
        else
            Ok ()
    
    override _.ExecuteBlockAsync(content: string, context: ExecutionContext) =
        task {
            try
                let command = content.Trim()
                logger.LogInformation(sprintf "Executing command: %s" command)
                
                // For now, we'll simulate command execution
                // In a full implementation, this would execute the actual command
                let output = sprintf "Executing command: %s\nCommand execution completed (simulated)" command
                
                return Ok output
            with
            | ex ->
                return Error (createError (sprintf "COMMAND block execution failed: %s" ex.Message) (Some ex.StackTrace))
        }
