namespace TarsEngine.FSharp.Core.Working.Metascript

open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Handler for TEXT blocks in metascripts.
/// </summary>
type TextBlockHandler(logger: ILogger<TextBlockHandler>) =
    inherit BlockHandlerBase(logger, "TEXT", 70)
    
    override _.ValidateBlock(content: string) =
        // Text blocks can be empty
        Ok ()
    
    override _.ExecuteBlockAsync(content: string, context: ExecutionContext) =
        task {
            try
                logger.LogDebug(sprintf "Processing text content (length: %d)" content.Length)
                
                // Process text content - could include variable substitution, formatting, etc.
                let processedContent = content.Trim()
                let output = sprintf "Text content:\n%s" processedContent
                
                return Ok output
            with
            | ex ->
                return Error (createError (sprintf "TEXT block processing failed: %s" ex.Message) (Some ex.StackTrace))
        }
