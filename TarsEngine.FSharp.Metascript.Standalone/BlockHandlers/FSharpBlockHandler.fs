namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System
open System.IO
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open FSharp.Compiler.Interactive.Shell
open TarsEngine.FSharp.Metascript

/// <summary>
/// Handler for F# blocks.
/// </summary>
type FSharpBlockHandler(logger: ILogger<FSharpBlockHandler>) =
    inherit BlockHandlerBase(logger, MetascriptBlockType.FSharp, 100)
    
    /// <summary>
    /// Creates a new F# interactive session.
    /// </summary>
    /// <returns>The F# interactive session.</returns>
    let createFsiSession() =
        let sbOut = new StringBuilder()
        let sbErr = new StringBuilder()
        let inStream = new StringReader("")
        let outStream = new StringWriter(sbOut)
        let errStream = new StringWriter(sbErr)
        
        let fsiConfig = FsiEvaluationSession.GetDefaultConfiguration()
        let fsiSession = FsiEvaluationSession.Create(fsiConfig, [|"--noninteractive"|], inStream, outStream, errStream)
        
        (fsiSession, sbOut, sbErr)
    
    /// <summary>
    /// Executes an F# block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    override this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            try
                // Create a new F# interactive session
                let (fsiSession, sbOut, sbErr) = createFsiSession()
                
                // Add variables from context
                for KeyValue(name, variable) in context.Variables do
                    fsiSession.EvalInteraction($"let {name} = {variable.Value}")
                
                // Execute the code
                let result = fsiSession.EvalExpression(block.Content)
                
                // Get the output
                let output = sbOut.ToString()
                let error = sbErr.ToString()
                
                // Get the return value
                let returnValue = 
                    match result with
                    | Some value -> Some (value.ReflectionValue)
                    | None -> None
                
                // Get the variables
                let variables = Map.empty<string, MetascriptVariable>
                
                // Check for errors
                if not (String.IsNullOrWhiteSpace(error)) then
                    return this.CreateFailureResult(block, error, output, variables)
                else
                    return this.CreateSuccessResult(block, output, returnValue, variables)
            with
            | ex ->
                logger.LogError(ex, "Error executing F# block")
                return this.CreateFailureResult(block, ex.ToString())
        }
