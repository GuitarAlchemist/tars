namespace TarsEngine.DSL

open System
open System.Collections.Generic
open Ast

/// <summary>
/// Represents the result of combining multiple parsed chunks.
/// </summary>
type CombinedParseResult = {
    /// <summary>
    /// The combined program from all chunks.
    /// </summary>
    Program: TarsProgram
    
    /// <summary>
    /// The combined diagnostics from all chunks.
    /// </summary>
    Diagnostics: Diagnostic list
    
    /// <summary>
    /// The individual chunk parse results.
    /// </summary>
    ChunkResults: ChunkParseResult list
    
    /// <summary>
    /// Unresolved references between chunks.
    /// </summary>
    UnresolvedReferences: string list
}

/// <summary>
/// Module for combining parsed chunks into a complete program.
/// </summary>
module ChunkCombiner =
    /// <summary>
    /// Combine multiple parsed chunks into a single program.
    /// </summary>
    /// <param name="chunkResults">The parsed chunk results to combine.</param>
    /// <returns>The combined parse result.</returns>
    let combineChunks (chunkResults: ChunkParseResult list) =
        // Sort chunks by index
        let sortedChunkResults = 
            chunkResults 
            |> List.sortBy (fun result -> result.Chunk.ChunkIndex)
        
        // Combine programs
        let combinedBlocks = 
            sortedChunkResults 
            |> List.collect (fun result -> result.Program.Blocks)
        
        let combinedProgram = { Blocks = combinedBlocks }
        
        // Combine diagnostics
        let combinedDiagnostics = 
            sortedChunkResults 
            |> List.collect (fun result -> 
                ChunkParser.adjustDiagnostics result.Chunk result.Diagnostics)
        
        // Find unresolved references
        let exportedBlocks = 
            sortedChunkResults 
            |> List.collect (fun result -> result.ExportedBlocks)
            |> Set.ofList
        
        let externalReferences = 
            sortedChunkResults 
            |> List.collect (fun result -> result.ExternalReferences)
            |> Set.ofList
        
        let unresolvedReferences = 
            Set.difference externalReferences exportedBlocks
            |> Set.toList
        
        // Create the result
        {
            Program = combinedProgram
            Diagnostics = combinedDiagnostics
            ChunkResults = sortedChunkResults
            UnresolvedReferences = unresolvedReferences
        }
    
    /// <summary>
    /// Validate the combined parse result.
    /// </summary>
    /// <param name="combinedResult">The combined parse result to validate.</param>
    /// <returns>A list of validation diagnostics.</returns>
    let validateCombinedResult (combinedResult: CombinedParseResult) =
        let validationDiagnostics = ResizeArray<Diagnostic>()
        
        // Check for unresolved references
        for reference in combinedResult.UnresolvedReferences do
            let diagnostic = {
                Severity = DiagnosticSeverity.Warning
                Code = WarningCode.UnusedProperty
                Message = sprintf "Unresolved reference to '%s'" reference
                Line = 1 // We don't know the exact line
                Column = 1 // We don't know the exact column
                LineContent = "" // We don't know the exact line content
                Suggestions = ["Check that the referenced block or variable is defined."]
            }
            
            validationDiagnostics.Add(diagnostic)
        
        // Check for duplicate block names
        let blockNames = Dictionary<string, int>()
        
        let rec checkBlock (block: TarsBlock) =
            match block.Name with
            | Some name ->
                if blockNames.ContainsKey(name) then
                    let diagnostic = {
                        Severity = DiagnosticSeverity.Warning
                        Code = WarningCode.DuplicateProperty
                        Message = sprintf "Duplicate block name '%s'" name
                        Line = 1 // We don't know the exact line
                        Column = 1 // We don't know the exact column
                        LineContent = "" // We don't know the exact line content
                        Suggestions = ["Rename one of the blocks to avoid conflicts."]
                    }
                    
                    validationDiagnostics.Add(diagnostic)
                else
                    blockNames.Add(name, 1)
            | None -> ()
            
            // Check nested blocks
            for nestedBlock in block.NestedBlocks do
                checkBlock nestedBlock
        
        // Check all blocks in the program
        for block in combinedResult.Program.Blocks do
            checkBlock block
        
        validationDiagnostics |> Seq.toList
