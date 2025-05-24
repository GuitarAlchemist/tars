namespace TarsEngine.FSharp.Core.Working.Metascript

open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Handler for CONFIG blocks in metascripts.
/// </summary>
type ConfigBlockHandler(logger: ILogger<ConfigBlockHandler>) =
    inherit BlockHandlerBase(logger, "CONFIG", 100)
    
    override _.ValidateBlock(content: string) =
        if System.String.IsNullOrWhiteSpace(content) then
            Error (createError "CONFIG block cannot be empty" None)
        else
            Ok ()
    
    override _.ExecuteBlockAsync(content: string, context: ExecutionContext) =
        task {
            try
                // Parse configuration key-value pairs
                let lines = content.Split([|'\n'; '\r'|], System.StringSplitOptions.RemoveEmptyEntries)
                let mutable variables = context.Variables
                
                for line in lines do
                    let trimmedLine = line.Trim()
                    if not (System.String.IsNullOrEmpty(trimmedLine)) && not (trimmedLine.StartsWith("//")) then
                        let parts = trimmedLine.Split([|':'|], 2)
                        if parts.Length = 2 then
                            let key = parts.[0].Trim()
                            let value = parts.[1].Trim().Trim([|'"'; '\''|])
                            variables <- Map.add key (box value) variables
                            logger.LogDebug(sprintf "Set configuration: %s = %s" key value)
                
                let output = sprintf "Configuration processed: %d variables set" (Map.count variables - Map.count context.Variables)
                return Ok output
            with
            | ex ->
                return Error (createError (sprintf "CONFIG block execution failed: %s" ex.Message) (Some ex.StackTrace))
        }
