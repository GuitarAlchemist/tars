namespace TarsEngine.DSL

open System.Collections.Generic
open Ast

/// <summary>
/// Module for handling deprecated property value warnings.
/// </summary>
module DeprecatedPropertyValueWarnings =
    /// <summary>
    /// Registry of deprecated property values with their alternatives.
    /// </summary>
    let private deprecatedPropertyValues = Dictionary<string, Dictionary<string, Dictionary<string, string option>>>()
    
    /// <summary>
    /// Initialize the deprecated property values registry with default values.
    /// </summary>
    let initialize() =
        // Clear existing registry
        deprecatedPropertyValues.Clear()
        
        // Add deprecated property values for each block type and property
        
        // CONFIG block deprecated property values
        let configProperties = Dictionary<string, Dictionary<string, string option>>()
        
        // CONFIG.api_version deprecated values
        let apiVersionValues = Dictionary<string, string option>()
        apiVersionValues.Add("1.0", Some "2.0") // api_version 1.0 is deprecated in favor of 2.0
        configProperties.Add("api_version", apiVersionValues)
        
        // CONFIG.mode deprecated values
        let modeValues = Dictionary<string, string option>()
        modeValues.Add("legacy", Some "standard") // mode legacy is deprecated in favor of standard
        configProperties.Add("mode", modeValues)
        
        deprecatedPropertyValues.Add("CONFIG", configProperties)
        
        // PROMPT block deprecated property values
        let promptProperties = Dictionary<string, Dictionary<string, string option>>()
        
        // PROMPT.content_type deprecated values
        let contentTypeValues = Dictionary<string, string option>()
        contentTypeValues.Add("text", Some "markdown") // content_type text is deprecated in favor of markdown
        promptProperties.Add("content_type", contentTypeValues)
        
        deprecatedPropertyValues.Add("PROMPT", promptProperties)
        
        // AGENT block deprecated property values
        let agentProperties = Dictionary<string, Dictionary<string, string option>>()
        
        // AGENT.agent_type deprecated values
        let agentTypeValues = Dictionary<string, string option>()
        agentTypeValues.Add("basic", Some "standard") // agent_type basic is deprecated in favor of standard
        agentProperties.Add("agent_type", agentTypeValues)
        
        deprecatedPropertyValues.Add("AGENT", agentProperties)
    
    // Initialize the registry
    do initialize()
    
    /// <summary>
    /// Check if a property value is deprecated for a specific block type and property.
    /// </summary>
    /// <param name="blockType">The block type containing the property.</param>
    /// <param name="propertyName">The property name containing the value.</param>
    /// <param name="propertyValue">The property value to check.</param>
    /// <returns>A tuple of (isDeprecated, alternativePropertyValue).</returns>
    let isDeprecated (blockType: string) (propertyName: string) (propertyValue: string) =
        match deprecatedPropertyValues.TryGetValue(blockType.ToUpper()) with
        | true, properties ->
            match properties.TryGetValue(propertyName.ToLower()) with
            | true, values ->
                match values.TryGetValue(propertyValue.ToLower()) with
                | true, alternative -> (true, alternative)
                | false, _ -> (false, None)
            | false, _ -> (false, None)
        | false, _ -> (false, None)
    
    /// <summary>
    /// Get all deprecated property values for a specific block type and property.
    /// </summary>
    /// <param name="blockType">The block type.</param>
    /// <param name="propertyName">The property name.</param>
    /// <returns>A list of all deprecated property values with their alternatives for the specified block type and property.</returns>
    let getDeprecatedPropertyValuesForProperty (blockType: string) (propertyName: string) =
        match deprecatedPropertyValues.TryGetValue(blockType.ToUpper()) with
        | true, properties ->
            match properties.TryGetValue(propertyName.ToLower()) with
            | true, values ->
                values
                |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
                |> Seq.toList
            | false, _ -> []
        | false, _ -> []
    
    /// <summary>
    /// Get all deprecated property values for all block types and properties.
    /// </summary>
    /// <returns>A list of tuples containing (blockType, propertyName, propertyValue, alternativePropertyValue).</returns>
    let getAllDeprecatedPropertyValues() =
        deprecatedPropertyValues
        |> Seq.collect (fun blockKvp -> 
            blockKvp.Value 
            |> Seq.collect (fun propKvp -> 
                propKvp.Value 
                |> Seq.map (fun valueKvp -> 
                    (blockKvp.Key, propKvp.Key, valueKvp.Key, valueKvp.Value))))
        |> Seq.toList
    
    /// <summary>
    /// Register a deprecated property value for a specific block type and property.
    /// </summary>
    /// <param name="blockType">The block type containing the property.</param>
    /// <param name="propertyName">The property name containing the value.</param>
    /// <param name="propertyValue">The deprecated property value.</param>
    /// <param name="alternativePropertyValue">The alternative property value to use instead, or None if there is no alternative.</param>
    let registerDeprecatedPropertyValue (blockType: string) (propertyName: string) (propertyValue: string) (alternativePropertyValue: string option) =
        let blockTypeUpper = blockType.ToUpper()
        let propertyNameLower = propertyName.ToLower()
        let propertyValueLower = propertyValue.ToLower()
        
        if not (deprecatedPropertyValues.ContainsKey(blockTypeUpper)) then
            deprecatedPropertyValues.Add(blockTypeUpper, Dictionary<string, Dictionary<string, string option>>())
        
        if not (deprecatedPropertyValues.[blockTypeUpper].ContainsKey(propertyNameLower)) then
            deprecatedPropertyValues.[blockTypeUpper].Add(propertyNameLower, Dictionary<string, string option>())
        
        deprecatedPropertyValues.[blockTypeUpper].[propertyNameLower].[propertyValueLower] <- alternativePropertyValue
    
    /// <summary>
    /// Unregister a deprecated property value for a specific block type and property.
    /// </summary>
    /// <param name="blockType">The block type containing the property.</param>
    /// <param name="propertyName">The property name containing the value.</param>
    /// <param name="propertyValue">The deprecated property value to unregister.</param>
    let unregisterDeprecatedPropertyValue (blockType: string) (propertyName: string) (propertyValue: string) =
        let blockTypeUpper = blockType.ToUpper()
        let propertyNameLower = propertyName.ToLower()
        let propertyValueLower = propertyValue.ToLower()
        
        if deprecatedPropertyValues.ContainsKey(blockTypeUpper) &&
           deprecatedPropertyValues.[blockTypeUpper].ContainsKey(propertyNameLower) then
            deprecatedPropertyValues.[blockTypeUpper].[propertyNameLower].Remove(propertyValueLower) |> ignore
    
    /// <summary>
    /// Check a block for deprecated property values and generate warnings if needed.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <param name="line">The line number where the block starts.</param>
    /// <param name="column">The column number where the block starts.</param>
    /// <param name="lineContent">The line content where the block starts.</param>
    /// <returns>A list of diagnostic messages for deprecated property values.</returns>
    let checkBlock (block: TarsBlock) line column lineContent =
        let blockType = 
            match block.Type with
            | BlockType.Unknown name -> name
            | _ -> block.Type.ToString()
        
        let warnings = ResizeArray<Diagnostic>()
        
        for KeyValue(propertyName, propertyValue) in block.Properties do
            match propertyValue with
            | StringValue value ->
                let (isDeprecated, alternative) = isDeprecated blockType propertyName value
                
                if isDeprecated && WarningRegistry.isEnabled WarningCode.DeprecatedPropertyValue then
                    warnings.Add(WarningGenerator.generateDeprecatedPropertyValueWarning blockType propertyName value alternative line column lineContent)
            | _ -> ()
        
        warnings |> Seq.toList
