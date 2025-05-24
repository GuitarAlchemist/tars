namespace TarsEngine.FSharp.Core.Working.Metascript

open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Handler for FSHARP blocks in metascripts.
/// </summary>
type FSharpBlockHandler(logger: ILogger<FSharpBlockHandler>) =
    inherit BlockHandlerBase(logger, "FSHARP", 90)
    
    override _.ValidateBlock(content: string) =
        if System.String.IsNullOrWhiteSpace(content) then
            Error (createError "FSHARP block cannot be empty" None)
        else
            Ok ()
    
    override _.ExecuteBlockAsync(content: string, context: ExecutionContext) =
        task {
            try
                logger.LogInformation("Executing F# code block")
                logger.LogDebug(sprintf "F# Code:\n%s" content)
                
                // For now, we'll simulate F# execution
                // In a full implementation, this would use FSharp.Compiler.Service
                let output = sprintf "F# code executed successfully\n// F# Code:\n%s\n// F# execution completed (simulated)" content
                
                return Ok output
            with
            | ex ->
                return Error (createError (sprintf "FSHARP block execution failed: %s" ex.Message) (Some ex.StackTrace))
        }
