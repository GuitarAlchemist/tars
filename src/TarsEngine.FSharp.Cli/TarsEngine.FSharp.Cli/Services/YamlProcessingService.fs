namespace TarsEngine.FSharp.Cli.Services

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging
open YamlDotNet.Serialization
open YamlDotNet.Serialization.NamingConventions

/// <summary>
/// Service for processing YAML content in metascripts.
/// </summary>
type YamlProcessingService(logger: ILogger<YamlProcessingService>) =
    
    let deserializer = 
        DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .Build()
    
    let serializer = 
        SerializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .Build()
    
    /// <summary>
    /// Process YAML content and extract variables.
    /// </summary>
    member this.ProcessYamlContent(yamlContent: string) =
        try
            logger.LogInformation("⚙️ TARS: Processing YAML configuration...")
            
            // Parse YAML content
            let yamlObject = deserializer.Deserialize<Dictionary<string, obj>>(yamlContent)
            
            let variables = Dictionary<string, obj>()
            
            // Extract variables from YAML
            this.extractVariablesFromYaml yamlObject "" variables
            
            logger.LogInformation("✅ TARS: YAML processing completed - extracted {Count} variables", variables.Count)
            
            // Convert to F# Map
            variables |> Seq.map (|KeyValue|) |> Map.ofSeq
            
        with
        | ex ->
            logger.LogError(ex, "❌ TARS: YAML processing failed: {Error}", ex.Message)
            Map.empty
    
    /// <summary>
    /// Recursively extract variables from YAML object.
    /// </summary>
    member private this.extractVariablesFromYaml (yamlObj: obj) (prefix: string) (variables: Dictionary<string, obj>) =
        match yamlObj with
        | :? Dictionary<string, obj> as dict ->
            for kvp in dict do
                let key = if String.IsNullOrEmpty(prefix) then kvp.Key else sprintf "%s.%s" prefix kvp.Key
                match kvp.Value with
                | :? Dictionary<string, obj> as nestedDict ->
                    this.extractVariablesFromYaml nestedDict key variables
                | :? System.Collections.IList as list ->
                    variables.[key] <- box (this.convertListToArray list)
                | value ->
                    variables.[key] <- value
        | :? System.Collections.IList as list ->
            variables.[prefix] <- box (this.convertListToArray list)
        | value ->
            variables.[prefix] <- value
    
    /// <summary>
    /// Convert YAML list to array.
    /// </summary>
    member private this.convertListToArray (list: System.Collections.IList) =
        let array = Array.zeroCreate list.Count
        for i = 0 to list.Count - 1 do
            array.[i] <- list.[i]
        array
    
    /// <summary>
    /// Serialize object to YAML.
    /// </summary>
    member this.SerializeToYaml(obj: obj) =
        try
            serializer.Serialize(obj)
        with
        | ex ->
            logger.LogError(ex, "Failed to serialize object to YAML")
            ""
    
    /// <summary>
    /// Validate YAML content.
    /// </summary>
    member this.ValidateYaml(yamlContent: string) =
        try
            deserializer.Deserialize<Dictionary<string, obj>>(yamlContent) |> ignore
            true
        with
        | _ -> false
