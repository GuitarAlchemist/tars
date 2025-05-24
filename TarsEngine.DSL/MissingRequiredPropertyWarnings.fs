namespace TarsEngine.DSL

open System.Collections.Generic
open Ast

/// <summary>
/// Module for handling missing required property warnings.
/// </summary>
module MissingRequiredPropertyWarnings =
    /// <summary>
    /// Registry of required properties for each block type.
    /// </summary>
    let private requiredProperties = Dictionary<string, string list>()
    
    /// <summary>
    /// Initialize the required properties registry with default values.
    /// </summary>
    let initialize() =
        // Clear existing registry
        requiredProperties.Clear()
        
        // Add required properties for each block type
        
        // CONFIG block required properties
        requiredProperties.Add("CONFIG", ["name"; "api_version"])
        
        // PROMPT block required properties
        requiredProperties.Add("PROMPT", ["content"])
        
        // AGENT block required properties
        requiredProperties.Add("AGENT", ["name"; "agent_type"])
        
        // VARIABLE block required properties
        requiredProperties.Add("VARIABLE", ["name"; "value"])
        
        // FUNCTION block required properties
        requiredProperties.Add("FUNCTION", ["name"; "parameters"])
        
        // TASK block required properties
        requiredProperties.Add("TASK", ["name"; "description"])
        
        // ACTION block required properties
        requiredProperties.Add("ACTION", ["name"; "description"])
        
        // TEMPLATE block required properties
        requiredProperties.Add("TEMPLATE", ["name"])
        
        // USE_TEMPLATE block required properties
        requiredProperties.Add("USE_TEMPLATE", ["template"])
    
    // Initialize the registry
    do initialize()
    
    /// <summary>
    /// Get the required properties for a specific block type.
    /// </summary>
    /// <param name="blockType">The block type.</param>
    /// <returns>A list of required property names for the specified block type.</returns>
    let getRequiredPropertiesForBlockType (blockType: string) =
        match requiredProperties.TryGetValue(blockType.ToUpper()) with
        | true, properties -> properties
        | false, _ -> []
    
    /// <summary>
    /// Get all required properties for all block types.
    /// </summary>
    /// <returns>A list of tuples containing (blockType, requiredProperties).</returns>
    let getAllRequiredProperties() =
        requiredProperties
        |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
        |> Seq.toList
    
    /// <summary>
    /// Register required properties for a specific block type.
    /// </summary>
    /// <param name="blockType">The block type.</param>
    /// <param name="properties">The list of required property names.</param>
    let registerRequiredProperties (blockType: string) (properties: string list) =
        requiredProperties.[blockType.ToUpper()] <- properties
    
    /// <summary>
    /// Unregister required properties for a specific block type.
    /// </summary>
    /// <param name="blockType">The block type to unregister.</param>
    let unregisterRequiredProperties (blockType: string) =
        requiredProperties.Remove(blockType.ToUpper()) |> ignore
    
    /// <summary>
    /// Check a block for missing required properties and generate warnings if needed.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <param name="line">The line number where the block starts.</param>
    /// <param name="column">The column number where the block starts.</param>
    /// <param name="lineContent">The line content where the block starts.</param>
    /// <returns>A list of diagnostic messages for missing required properties.</returns>
    let checkBlock (block: TarsBlock) line column lineContent =
        let blockType = 
            match block.Type with
            | BlockType.Unknown name -> name
            | _ -> block.Type.ToString()
        
        let warnings = ResizeArray<Diagnostic>()
        
        let requiredProps = getRequiredPropertiesForBlockType blockType
        
        for requiredProp in requiredProps do
            if not (block.Properties.ContainsKey(requiredProp)) && WarningRegistry.isEnabled WarningCode.MissingRequiredProperty then
                warnings.Add(WarningGenerator.generateMissingRequiredPropertyWarning blockType requiredProp line column lineContent)
        
        warnings |> Seq.toList
