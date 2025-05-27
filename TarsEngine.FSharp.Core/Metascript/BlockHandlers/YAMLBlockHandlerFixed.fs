namespace TarsEngine.FSharp.Core.Metascript.BlockHandlers

open System
open System.Collections.Generic
open System.Text.Json
open TarsEngine.FSharp.Core.Metascript.Types

/// <summary>
/// Handler for YAML blocks in metascripts
/// Provides YAML processing and status management capabilities
/// </summary>
module YAMLBlockHandler =
    
    /// <summary>
    /// Parses YAML content into a structured format
    /// </summary>
    let parseYamlContent (content: string) (context: MetascriptExecutionContext) : Result<Map<string, obj>, string> =
        try
            // Simple YAML parser (for basic key-value pairs)
            let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
            let mutable yamlData = Map.empty<string, obj>
            
            for line in lines do
                let trimmedLine = line.Trim()
                if not (String.IsNullOrEmpty(trimmedLine)) && not (trimmedLine.StartsWith("#")) then
                    let colonIndex = trimmedLine.IndexOf(':')
                    if colonIndex > 0 then
                        let key = trimmedLine.Substring(0, colonIndex).Trim()
                        let value = trimmedLine.Substring(colonIndex + 1).Trim()
                        yamlData <- yamlData.Add(key, value :> obj)
            
            Ok yamlData
            
        with
        | ex -> Error ex.Message
    
    /// <summary>
    /// Processes a status section
    /// </summary>
    let processStatusSection (status: string) (context: MetascriptExecutionContext) : string =
        sprintf "Status updated: %s" status
    
    /// <summary>
    /// Processes a phase section
    /// </summary>
    let processPhaseSection (phase: string) (context: MetascriptExecutionContext) : string =
        sprintf "Phase set: %s" phase
    
    /// <summary>
    /// Processes a progress section
    /// </summary>
    let processProgressSection (progress: string) (context: MetascriptExecutionContext) : string =
        sprintf "Progress updated: %s" progress
    
    /// <summary>
    /// Processes an exploration section
    /// </summary>
    let processExplorationSection (exploration: string) (context: MetascriptExecutionContext) : string =
        sprintf "Exploration mode: %s" exploration
    
    /// <summary>
    /// Processes a recovery section
    /// </summary>
    let processRecoverySection (recovery: string) (context: MetascriptExecutionContext) : string =
        sprintf "Recovery action: %s" recovery
    
    /// <summary>
    /// Processes parsed YAML data
    /// </summary>
    let processYamlData (yamlData: Map<string, obj>) (context: MetascriptExecutionContext) : string =
        let mutable output = []
        
        // Process different YAML structures
        for kvp in yamlData do
            match kvp.Key.ToLower() with
            | "status" -> 
                output <- output @ [processStatusSection (kvp.Value.ToString()) context]
            | "phase" ->
                output <- output @ [processPhaseSection (kvp.Value.ToString()) context]
            | "progress" ->
                output <- output @ [processProgressSection (kvp.Value.ToString()) context]
            | "exploration" ->
                output <- output @ [processExplorationSection (kvp.Value.ToString()) context]
            | "recovery" ->
                output <- output @ [processRecoverySection (kvp.Value.ToString()) context]
            | _ ->
                output <- output @ [sprintf "Processed YAML key: %s = %s" kvp.Key (kvp.Value.ToString())]
        
        String.Join("\n", output)
    
    /// <summary>
    /// Generates YAML content from a status map
    /// </summary>
    let generateYamlContent (status: Map<string, obj>) : string =
        let mutable lines = []
        
        for kvp in status do
            lines <- lines @ [sprintf "%s: %s" kvp.Key (kvp.Value.ToString())]
        
        String.Join("\n", lines)
    
    /// <summary>
    /// Creates a YAML status file
    /// </summary>
    let createStatusFile (filePath: string) (status: Map<string, obj>) : Result<string, string> =
        try
            let yamlContent = generateYamlContent status
            System.IO.File.WriteAllText(filePath, yamlContent)
            Ok (sprintf "Status file created: %s" filePath)
        with
        | ex -> Error ex.Message
    
    /// <summary>
    /// Updates an existing YAML status file
    /// </summary>
    let updateStatusFile (filePath: string) (updates: Map<string, obj>) : Result<string, string> =
        try
            // Read existing content if file exists
            let existingContent = 
                if System.IO.File.Exists(filePath) then
                    System.IO.File.ReadAllText(filePath)
                else
                    ""
            
            // Parse existing YAML
            let existingData = 
                match parseYamlContent existingContent { Variables = Map.empty; ProjectPath = ""; OutputPath = "" } with
                | Ok data -> data
                | Error _ -> Map.empty
            
            // Merge with updates
            let mergedData = 
                updates |> Map.fold (fun acc key value -> acc.Add(key, value)) existingData
            
            // Generate new YAML content
            let newContent = generateYamlContent mergedData
            System.IO.File.WriteAllText(filePath, newContent)
            
            Ok (sprintf "Status file updated: %s" filePath)
        with
        | ex -> Error ex.Message
    
    /// <summary>
    /// Executes a YAML block with status management capabilities
    /// </summary>
    /// <param name="content">The YAML block content</param>
    /// <param name="context">The execution context</param>
    /// <returns>The execution result</returns>
    let executeYamlBlock (content: string) (context: MetascriptExecutionContext) : MetascriptBlockResult =
        try
            // Initialize result
            let mutable result = {
                Success = true
                Output = ""
                Variables = Map.empty
                Logs = []
                ExecutionTime = TimeSpan.Zero
            }
            
            let startTime = DateTime.Now
            
            // Parse YAML content
            let yamlResult = parseYamlContent content context
            
            match yamlResult with
            | Ok yamlData ->
                // Process the YAML data
                let processedResult = processYamlData yamlData context
                result <- 
                    { result with 
                        Output = processedResult
                        Logs = result.Logs @ ["YAML block processed successfully"] }
            
            | Error errorMsg ->
                result <- 
                    { result with 
                        Success = false
                        Output = sprintf "YAML parsing error: %s" errorMsg
                        Logs = result.Logs @ [sprintf "YAML parsing failed: %s" errorMsg] }
            
            let endTime = DateTime.Now
            { result with ExecutionTime = endTime - startTime }
            
        with
        | ex ->
            {
                Success = false
                Output = sprintf "Error executing YAML block: %s" ex.Message
                Variables = Map.empty
                Logs = [sprintf "YAML block execution failed: %s" ex.Message]
                ExecutionTime = TimeSpan.Zero
            }
