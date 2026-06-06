namespace TarsEngine.DSL

open System.Collections.Generic
open Ast

/// <summary>
/// Module for handling deprecated block type warnings.
/// </summary>
module DeprecatedBlockTypeWarnings =
    /// <summary>
    /// Registry of deprecated block types with their alternatives.
    /// </summary>
    let private deprecatedBlockTypes = Dictionary<string, string option>()
    
    /// <summary>
    /// Initialize the deprecated block types registry with default values.
    /// </summary>
    let initialize() =
        // Clear existing registry
        deprecatedBlockTypes.Clear()
        
        // Add deprecated block types
        // Format: deprecatedBlockTypes.Add("DEPRECATED_TYPE", Some "ALTERNATIVE_TYPE")
        // or: deprecatedBlockTypes.Add("DEPRECATED_TYPE", None)
        
        // Example: DESCRIBE is deprecated in favor of PROMPT
        deprecatedBlockTypes.Add("DESCRIBE", Some "PROMPT")
        
        // Example: SPAWN_AGENT is deprecated in favor of AGENT
        deprecatedBlockTypes.Add("SPAWN_AGENT", Some "AGENT")
        
        // Example: SELF_IMPROVE is deprecated in favor of AUTO_IMPROVE
        deprecatedBlockTypes.Add("SELF_IMPROVE", Some "AUTO_IMPROVE")
    
    // Initialize the registry
    do initialize()
    
    /// <summary>
    /// Check if a block type is deprecated.
    /// </summary>
    /// <param name="blockType">The block type to check.</param>
    /// <returns>A tuple of (isDeprecated, alternativeBlockType).</returns>
    let isDeprecated (blockType: string) =
        match deprecatedBlockTypes.TryGetValue(blockType.ToUpper()) with
        | true, alternative -> (true, alternative)
        | false, _ -> (false, None)
    
    /// <summary>
    /// Get all deprecated block types.
    /// </summary>
    /// <returns>A list of all deprecated block types with their alternatives.</returns>
    let getAllDeprecatedBlockTypes() =
        deprecatedBlockTypes
        |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
        |> Seq.toList
    
    /// <summary>
    /// Register a deprecated block type.
    /// </summary>
    /// <param name="blockType">The deprecated block type.</param>
    /// <param name="alternativeBlockType">The alternative block type to use instead, or None if there is no alternative.</param>
    let registerDeprecatedBlockType (blockType: string) (alternativeBlockType: string option) =
        deprecatedBlockTypes.[blockType.ToUpper()] <- alternativeBlockType
    
    /// <summary>
    /// Unregister a deprecated block type.
    /// </summary>
    /// <param name="blockType">The deprecated block type to unregister.</param>
    let unregisterDeprecatedBlockType (blockType: string) =
        deprecatedBlockTypes.Remove(blockType.ToUpper()) |> ignore
    
    /// <summary>
    /// Check a block for deprecated block type and generate a warning if needed.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <param name="line">The line number where the block starts.</param>
    /// <param name="column">The column number where the block starts.</param>
    /// <param name="lineContent">The line content where the block starts.</param>
    /// <returns>A list of diagnostic messages for deprecated block types.</returns>
    let checkBlock (block: TarsBlock) line column lineContent =
        let blockType = 
            match block.Type with
            | BlockType.Unknown name -> name
            | _ -> block.Type.ToString()
        
        let (isDeprecated, alternative) = isDeprecated blockType
        
        if isDeprecated && WarningRegistry.isEnabled WarningCode.DeprecatedBlockType then
            [WarningGenerator.generateDeprecatedBlockTypeWarning blockType alternative line column lineContent]
        else
            []
