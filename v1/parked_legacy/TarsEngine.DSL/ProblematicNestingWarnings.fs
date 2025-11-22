namespace TarsEngine.DSL

open System.Collections.Generic
open Ast

/// <summary>
/// Module for handling problematic nesting pattern warnings.
/// </summary>
module ProblematicNestingWarnings =
    /// <summary>
    /// Registry of problematic nesting patterns with their reasons.
    /// </summary>
    let private problematicNestingPatterns = Dictionary<string, Dictionary<string, string>>()
    
    /// <summary>
    /// Initialize the problematic nesting patterns registry with default values.
    /// </summary>
    let initialize() =
        // Clear existing registry
        problematicNestingPatterns.Clear()
        
        // Add problematic nesting patterns for each parent block type
        
        // CONFIG block problematic nesting patterns
        let configPatterns = Dictionary<string, string>()
        configPatterns.Add("CONFIG", "Nesting CONFIG blocks inside CONFIG blocks is not recommended.")
        configPatterns.Add("PROMPT", "Nesting PROMPT blocks inside CONFIG blocks is not recommended.")
        problematicNestingPatterns.Add("CONFIG", configPatterns)
        
        // PROMPT block problematic nesting patterns
        let promptPatterns = Dictionary<string, string>()
        promptPatterns.Add("CONFIG", "Nesting CONFIG blocks inside PROMPT blocks is not recommended.")
        problematicNestingPatterns.Add("PROMPT", promptPatterns)
        
        // VARIABLE block problematic nesting patterns
        let variablePatterns = Dictionary<string, string>()
        variablePatterns.Add("VARIABLE", "Nesting VARIABLE blocks inside VARIABLE blocks is not recommended.")
        problematicNestingPatterns.Add("VARIABLE", variablePatterns)
        
        // FUNCTION block problematic nesting patterns
        let functionPatterns = Dictionary<string, string>()
        functionPatterns.Add("FUNCTION", "Nesting FUNCTION blocks inside FUNCTION blocks is not recommended.")
        problematicNestingPatterns.Add("FUNCTION", functionPatterns)
    
    // Initialize the registry
    do initialize()
    
    /// <summary>
    /// Check if a nesting pattern is problematic.
    /// </summary>
    /// <param name="parentBlockType">The parent block type.</param>
    /// <param name="childBlockType">The child block type.</param>
    /// <returns>A tuple of (isProblematic, reason).</returns>
    let isProblematic (parentBlockType: string) (childBlockType: string) =
        match problematicNestingPatterns.TryGetValue(parentBlockType.ToUpper()) with
        | true, patterns ->
            match patterns.TryGetValue(childBlockType.ToUpper()) with
            | true, reason -> (true, reason)
            | false, _ -> (false, "")
        | false, _ -> (false, "")
    
    /// <summary>
    /// Get all problematic nesting patterns for a specific parent block type.
    /// </summary>
    /// <param name="parentBlockType">The parent block type.</param>
    /// <returns>A list of all problematic child block types with their reasons for the specified parent block type.</returns>
    let getProblematicNestingPatternsForParentBlockType (parentBlockType: string) =
        match problematicNestingPatterns.TryGetValue(parentBlockType.ToUpper()) with
        | true, patterns ->
            patterns
            |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
            |> Seq.toList
        | false, _ -> []
    
    /// <summary>
    /// Get all problematic nesting patterns for all parent block types.
    /// </summary>
    /// <returns>A list of tuples containing (parentBlockType, childBlockType, reason).</returns>
    let getAllProblematicNestingPatterns() =
        problematicNestingPatterns
        |> Seq.collect (fun parentKvp -> 
            parentKvp.Value 
            |> Seq.map (fun childKvp -> 
                (parentKvp.Key, childKvp.Key, childKvp.Value)))
        |> Seq.toList
    
    /// <summary>
    /// Register a problematic nesting pattern.
    /// </summary>
    /// <param name="parentBlockType">The parent block type.</param>
    /// <param name="childBlockType">The child block type.</param>
    /// <param name="reason">The reason why the nesting is problematic.</param>
    let registerProblematicNestingPattern (parentBlockType: string) (childBlockType: string) (reason: string) =
        let parentBlockTypeUpper = parentBlockType.ToUpper()
        let childBlockTypeUpper = childBlockType.ToUpper()
        
        if not (problematicNestingPatterns.ContainsKey(parentBlockTypeUpper)) then
            problematicNestingPatterns.Add(parentBlockTypeUpper, Dictionary<string, string>())
        
        problematicNestingPatterns.[parentBlockTypeUpper].[childBlockTypeUpper] <- reason
    
    /// <summary>
    /// Unregister a problematic nesting pattern.
    /// </summary>
    /// <param name="parentBlockType">The parent block type.</param>
    /// <param name="childBlockType">The child block type.</param>
    let unregisterProblematicNestingPattern (parentBlockType: string) (childBlockType: string) =
        let parentBlockTypeUpper = parentBlockType.ToUpper()
        let childBlockTypeUpper = childBlockType.ToUpper()
        
        if problematicNestingPatterns.ContainsKey(parentBlockTypeUpper) then
            problematicNestingPatterns.[parentBlockTypeUpper].Remove(childBlockTypeUpper) |> ignore
    
    /// <summary>
    /// Check a block for problematic nesting patterns and generate warnings if needed.
    /// </summary>
    /// <param name="parentBlock">The parent block.</param>
    /// <param name="line">The line number where the parent block starts.</param>
    /// <param name="column">The column number where the parent block starts.</param>
    /// <param name="lineContent">The line content where the parent block starts.</param>
    /// <returns>A list of diagnostic messages for problematic nesting patterns.</returns>
    let checkBlock (parentBlock: TarsBlock) line column lineContent =
        let parentBlockType = 
            match parentBlock.Type with
            | BlockType.Unknown name -> name
            | _ -> parentBlock.Type.ToString()
        
        let warnings = ResizeArray<Diagnostic>()
        
        for childBlock in parentBlock.NestedBlocks do
            let childBlockType = 
                match childBlock.Type with
                | BlockType.Unknown name -> name
                | _ -> childBlock.Type.ToString()
            
            let (isProblematic, reason) = isProblematic parentBlockType childBlockType
            
            if isProblematic && WarningRegistry.isEnabled WarningCode.ProblematicNesting then
                warnings.Add(WarningGenerator.generateProblematicNestingWarning parentBlockType childBlockType reason line column lineContent)
        
        warnings |> Seq.toList
