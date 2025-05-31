namespace TarsEngine.FSharp.Cli.Services

open System
open System.IO
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open FSharp.Compiler.Interactive.Shell

/// <summary>
/// Service for executing F# code using F# Interactive.
/// </summary>
type FSharpInteractiveService(logger: ILogger<FSharpInteractiveService>) =
    
    let mutable fsiSession: FsiEvaluationSession option = None
    
    /// <summary>
    /// Initialize F# Interactive session.
    /// </summary>
    member this.InitializeAsync() =
        task {
            try
                logger.LogInformation("🔧 TARS: Initializing F# Interactive session...")
                
                // Create F# Interactive configuration
                let fsiConfig = FsiEvaluationSession.GetDefaultConfiguration()
                let argv = [| "fsi.exe"; "--noninteractive"; "--nologo" |]
                let allArgs = Array.append argv [||]
                
                // Create input/output streams
                let inStream = new StringReader("")
                let outStream = new StringBuilder()
                let errStream = new StringBuilder()
                
                // Create F# Interactive session
                let session = FsiEvaluationSession.Create(fsiConfig, allArgs, inStream, new StringWriter(outStream), new StringWriter(errStream))
                fsiSession <- Some session
                
                logger.LogInformation("✅ TARS: F# Interactive session initialized successfully")
                return Ok ()
            with
            | ex ->
                let error = sprintf "Failed to initialize F# Interactive: %s" ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Execute F# code and return results.
    /// </summary>
    member this.ExecuteFSharpCodeAsync(code: string, variables: Map<string, obj>) =
        task {
            try
                match fsiSession with
                | None ->
                    let! initResult = this.InitializeAsync()
                    match initResult with
                    | Error err ->
                        return { Success = false; Output = ""; Error = Some err; Variables = variables }
                    | Ok _ ->
                        () // Continue execution
                | Some _ ->
                    () // Session already initialized
                
                match fsiSession with
                | Some session ->
                    logger.LogInformation("💻 TARS: Executing F# code block...")
                    
                    let output = StringBuilder()
                    let mutable newVariables = variables
                    
                    // Set up variables in F# Interactive
                    for kvp in variables do
                        try
                            let varCode = sprintf "let %s = %A" kvp.Key kvp.Value
                            let _, errors = session.EvalInteractionNonThrowing(varCode)
                            if errors.Length > 0 then
                                logger.LogWarning("Warning setting variable {Variable}: {Errors}", kvp.Key, String.concat "; " (errors |> Array.map (fun e -> e.Message)))
                        with
                        | ex -> logger.LogWarning(ex, "Failed to set variable {Variable}", kvp.Key)
                    
                    // Execute the F# code
                    try
                        let result, errors = session.EvalInteractionNonThrowing(code)

                        if errors.Length > 0 then
                            let errorMsg = String.concat "; " (errors |> Array.map (fun e -> e.Message))
                            logger.LogError("F# execution errors: {Errors}", errorMsg)
                            return {
                                Success = false
                                Output = output.ToString()
                                Error = Some errorMsg
                                Variables = variables
                            }
                        else
                            // Capture any output
                            match result with
                            | Choice1Of2 (Some value) ->
                                output.AppendLine(sprintf "Result: %A" value.ReflectionValue) |> ignore
                            | Choice1Of2 None ->
                                output.AppendLine("F# code executed successfully (no return value)") |> ignore
                            | Choice2Of2 ex ->
                                output.AppendLine(sprintf "Execution completed with exception: %s" ex.Message) |> ignore
                        
                        // Extract new variables (simplified - in a full implementation would use reflection)
                        // For now, we'll parse let bindings from the code
                        newVariables <- this.extractVariablesFromCode code variables
                        
                        logger.LogInformation("✅ TARS: F# code executed successfully")
                        return {
                            Success = true
                            Output = output.ToString()
                            Error = None
                            Variables = newVariables
                        }
                | None ->
                    return {
                        Success = false
                        Output = ""
                        Error = Some "F# Interactive session not available"
                        Variables = variables
                    }
            with
            | ex ->
                let error = sprintf "F# execution failed: %s" ex.Message
                logger.LogError(ex, error)
                return {
                    Success = false
                    Output = ""
                    Error = Some error
                    Variables = variables
                }
        }
    
    /// <summary>
    /// Extract variables from F# code (simplified implementation).
    /// </summary>
    member private this.extractVariablesFromCode (code: string) (existingVariables: Map<string, obj>) =
        let mutable variables = existingVariables
        
        // Simple regex-based extraction for let bindings
        let lines = code.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        for line in lines do
            let trimmed = line.Trim()
            if trimmed.StartsWith("let ") && trimmed.Contains("=") then
                try
                    let parts = trimmed.Substring(4).Split([|'='|], 2)
                    if parts.Length = 2 then
                        let varName = parts.[0].Trim()
                        let varValue = parts.[1].Trim().Trim('"')
                        variables <- Map.add varName (box varValue) variables
                with
                | _ -> () // Ignore parsing errors
        
        variables
    
    /// <summary>
    /// Dispose F# Interactive session.
    /// </summary>
    member this.Dispose() =
        match fsiSession with
        | Some session ->
            try
                (session :> IDisposable).Dispose()
                fsiSession <- None
                logger.LogInformation("🔧 TARS: F# Interactive session disposed")
            with
            | ex -> logger.LogWarning(ex, "Error disposing F# Interactive session")
        | None -> ()

/// <summary>
/// Execution result for F# code.
/// </summary>
and FSharpExecutionResult = {
    Success: bool
    Output: string
    Error: string option
    Variables: Map<string, obj>
}
