namespace TarsEngine.DSL

open System.Collections.Generic
open Ast

/// <summary>
/// Module for handling deprecated property warnings.
/// </summary>
module DeprecatedPropertyWarnings =
    /// <summary>
    /// Registry of deprecated properties with their alternatives.
    /// </summary>
    let private deprecatedProperties = Dictionary<string, Dictionary<string, string option>>()
    
    /// <summary>
    /// Initialize the deprecated properties registry with default values.
    /// </summary>
    let initialize() =
        // Clear existing registry
        deprecatedProperties.Clear()
        
        // Add deprecated properties for each block type
        
        // CONFIG block deprecated properties
        let configProperties = Dictionary<string, string option>()
        configProperties.Add("version", Some "api_version") // version is deprecated in favor of api_version
        configProperties.Add("author", Some "created_by") // author is deprecated in favor of created_by
        deprecatedProperties.Add("CONFIG", configProperties)
        
        // PROMPT block deprecated properties
        let promptProperties = Dictionary<string, string option>()
        promptProperties.Add("text", Some "content") // text is deprecated in favor of content
        promptProperties.Add("format", Some "content_type") // format is deprecated in favor of content_type
        deprecatedProperties.Add("PROMPT", promptProperties)
        
        // AGENT block deprecated properties
        let agentProperties = Dictionary<string, string option>()
        agentProperties.Add("type", Some "agent_type") // type is deprecated in favor of agent_type
        agentProperties.Add("config", Some "configuration") // config is deprecated in favor of configuration
        deprecatedProperties.Add("AGENT", agentProperties)
        
        // VARIABLE block deprecated properties
        let variableProperties = Dictionary<string, string option>()
        variableProperties.Add("type", Some "variable_type") // type is deprecated in favor of variable_type
        variableProperties.Add("default", Some "default_value") // default is deprecated in favor of default_value
        deprecatedProperties.Add("VARIABLE", variableProperties)
    
    // Initialize the registry
    do initialize()
    
    /// <summary>
    /// Check if a property is deprecated for a specific block type.
    /// </summary>
    /// <param name="blockType">The block type containing the property.</param>
    /// <param name="propertyName">The property name to check.</param>
    /// <returns>A tuple of (isDeprecated, alternativePropertyName).</returns>
    let isDeprecated (blockType: string) (propertyName: string) =
        match deprecatedProperties.TryGetValue(blockType.ToUpper()) with
        | true, properties ->
            match properties.TryGetValue(propertyName.ToLower()) with
            | true, alternative -> (true, alternative)
            | false, _ -> (false, None)
        | false, _ -> (false, None)
    
    /// <summary>
    /// Get all deprecated properties for a specific block type.
    /// </summary>
    /// <param name="blockType">The block type.</param>
    /// <returns>A list of all deprecated properties with their alternatives for the specified block type.</returns>
    let getDeprecatedPropertiesForBlockType (blockType: string) =
        match deprecatedProperties.TryGetValue(blockType.ToUpper()) with
        | true, properties ->
            properties
            |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
            |> Seq.toList
        | false, _ -> []
    
    /// <summary>
    /// Get all deprecated properties for all block types.
    /// </summary>
    /// <returns>A list of tuples containing (blockType, propertyName, alternativePropertyName).</returns>
    let getAllDeprecatedProperties() =
        deprecatedProperties
        |> Seq.collect (fun kvp -> 
            kvp.Value 
            |> Seq.map (fun prop -> (kvp.Key, prop.Key, prop.Value)))
        |> Seq.toList
    
    /// <summary>
    /// Register a deprecated property for a specific block type.
    /// </summary>
    /// <param name="blockType">The block type containing the property.</param>
    /// <param name="propertyName">The deprecated property name.</param>
    /// <param name="alternativePropertyName">The alternative property name to use instead, or None if there is no alternative.</param>
    let registerDeprecatedProperty (blockType: string) (propertyName: string) (alternativePropertyName: string option) =
        let blockTypeUpper = blockType.ToUpper()
        let propertyNameLower = propertyName.ToLower()
        
        if not (deprecatedProperties.ContainsKey(blockTypeUpper)) then
            deprecatedProperties.Add(blockTypeUpper, Dictionary<string, string option>())
        
        deprecatedProperties.[blockTypeUpper].[propertyNameLower] <- alternativePropertyName
    
    /// <summary>
    /// Unregister a deprecated property for a specific block type.
    /// </summary>
    /// <param name="blockType">The block type containing the property.</param>
    /// <param name="propertyName">The deprecated property name to unregister.</param>
    let unregisterDeprecatedProperty (blockType: string) (propertyName: string) =
        let blockTypeUpper = blockType.ToUpper()
        let propertyNameLower = propertyName.ToLower()
        
        if deprecatedProperties.ContainsKey(blockTypeUpper) then
            deprecatedProperties.[blockTypeUpper].Remove(propertyNameLower) |> ignore
    
    /// <summary>
    /// Check a block for deprecated properties and generate warnings if needed.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <param name="line">The line number where the block starts.</param>
    /// <param name="column">The column number where the block starts.</param>
    /// <param name="lineContent">The line content where the block starts.</param>
    /// <returns>A list of diagnostic messages for deprecated properties.</returns>
    let checkBlock (block: TarsBlock) line column lineContent =
        let blockType = 
            match block.Type with
            | BlockType.Unknown name -> name
            | _ -> block.Type.ToString()
        
        let warnings = ResizeArray<Diagnostic>()
        
        for KeyValue(propertyName, propertyValue) in block.Properties do
            let (isDeprecated, alternative) = isDeprecated blockType propertyName
            
            if isDeprecated && WarningRegistry.isEnabled WarningCode.DeprecatedProperty then
                warnings.Add(WarningGenerator.generateDeprecatedPropertyWarning blockType propertyName alternative line column lineContent)
        
        warnings |> Seq.toList
