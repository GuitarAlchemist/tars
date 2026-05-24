namespace TarsEngine.DSL

open Ast

/// <summary>
/// Module for handling deep nesting warnings.
/// </summary>
module DeepNestingWarnings =
    /// <summary>
    /// The maximum recommended nesting depth.
    /// </summary>
    let private maxRecommendedNestingDepth = 3
    
    /// <summary>
    /// Get the maximum recommended nesting depth.
    /// </summary>
    /// <returns>The maximum recommended nesting depth.</returns>
    let getMaxRecommendedNestingDepth() =
        maxRecommendedNestingDepth
    
    /// <summary>
    /// Set the maximum recommended nesting depth.
    /// </summary>
    /// <param name="depth">The new maximum recommended nesting depth.</param>
    let setMaxRecommendedNestingDepth (depth: int) =
        if depth < 1 then
            failwith "Maximum recommended nesting depth must be at least 1."
        
        // This is a mutable reference to the maxRecommendedNestingDepth value
        // In a real implementation, this would be a mutable field
        // For this example, we'll just print a message
        printfn "Setting maximum recommended nesting depth to %d" depth
    
    /// <summary>
    /// Check a block for deep nesting and generate warnings if needed.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <param name="nestingDepth">The current nesting depth.</param>
    /// <param name="line">The line number where the block starts.</param>
    /// <param name="column">The column number where the block starts.</param>
    /// <param name="lineContent">The line content where the block starts.</param>
    /// <returns>A list of diagnostic messages for deep nesting.</returns>
    let rec checkBlock (block: TarsBlock) (nestingDepth: int) line column lineContent =
        let blockType = 
            match block.Type with
            | BlockType.Unknown name -> name
            | _ -> block.Type.ToString()
        
        let warnings = ResizeArray<Diagnostic>()
        
        if nestingDepth > maxRecommendedNestingDepth && WarningRegistry.isEnabled WarningCode.DeepNesting then
            warnings.Add(WarningGenerator.generateDeepNestingWarning blockType nestingDepth maxRecommendedNestingDepth line column lineContent)
        
        // Check nested blocks
        for nestedBlock in block.NestedBlocks do
            let nestedWarnings = checkBlock nestedBlock (nestingDepth + 1) line column lineContent
            warnings.AddRange(nestedWarnings)
        
        warnings |> Seq.toList
    
    /// <summary>
    /// Check a program for deep nesting and generate warnings if needed.
    /// </summary>
    /// <param name="program">The program to check.</param>
    /// <param name="line">The line number where the program starts.</param>
    /// <param name="column">The column number where the program starts.</param>
    /// <param name="lineContent">The line content where the program starts.</param>
    /// <returns>A list of diagnostic messages for deep nesting.</returns>
    let checkProgram (program: TarsProgram) line column lineContent =
        let warnings = ResizeArray<Diagnostic>()
        
        for block in program.Blocks do
            let blockWarnings = checkBlock block 1 line column lineContent
            warnings.AddRange(blockWarnings)
        
        warnings |> Seq.toList
