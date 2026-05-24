namespace TarsEngine.DSL

open System
open System.Collections.Generic
open Ast

/// <summary>
/// Represents the result of parsing a chunk.
/// </summary>
type ChunkParseResult = {
    /// <summary>
    /// The chunk that was parsed.
    /// </summary>
    Chunk: CodeChunk
    
    /// <summary>
    /// The parsed program from the chunk.
    /// </summary>
    Program: TarsProgram
    
    /// <summary>
    /// The diagnostics generated during parsing.
    /// </summary>
    Diagnostics: Diagnostic list
    
    /// <summary>
    /// References to blocks in other chunks.
    /// </summary>
    ExternalReferences: string list
    
    /// <summary>
    /// Blocks that are defined in this chunk and may be referenced by other chunks.
    /// </summary>
    ExportedBlocks: string list
}

/// <summary>
/// Module for parsing individual chunks of code.
/// </summary>
module ChunkParser =
    /// <summary>
    /// Parse a single chunk of code.
    /// </summary>
    /// <param name="chunk">The chunk to parse.</param>
    /// <param name="parser">The parser function to use.</param>
    /// <returns>The result of parsing the chunk.</returns>
    let parseChunk (chunk: CodeChunk) (parser: string -> TarsProgram) =
        // Parse the chunk
        let program = parser chunk.Content
        
        // Collect diagnostics
        let diagnostics = ResizeArray<Diagnostic>()
        
        // Collect external references
        let externalReferences = ResizeArray<string>()
        
        // Collect exported blocks
        let exportedBlocks = ResizeArray<string>()
        
        // Process blocks to find references and exports
        let rec processBlock (block: TarsBlock) =
            // Check if the block has a name
            match block.Name with
            | Some name -> exportedBlocks.Add(name)
            | None -> ()
            
            // Check for references in properties
            for KeyValue(_, value) in block.Properties do
                match value with
                | StringValue str ->
                    // Check if the string contains a reference to another block
                    if str.Contains("@") then
                        let referencedBlocks = 
                            str.Split([|' '; '\t'; '\n'; '\r'; ','; '.'; ';'; '('; ')'; '['; ']'; '{'; '}'; '"'; '\''|], StringSplitOptions.RemoveEmptyEntries)
                            |> Array.filter (fun s -> s.StartsWith("@"))
                            |> Array.map (fun s -> s.Substring(1))
                        
                        for referencedBlock in referencedBlocks do
                            externalReferences.Add(referencedBlock)
                | VariableReference name -> externalReferences.Add(name)
                | _ -> ()
            
            // Process nested blocks
            for nestedBlock in block.NestedBlocks do
                processBlock nestedBlock
        
        // Process all blocks in the program
        for block in program.Blocks do
            processBlock block
        
        // Create the result
        {
            Chunk = chunk
            Program = program
            Diagnostics = diagnostics |> Seq.toList
            ExternalReferences = externalReferences |> Seq.distinct |> Seq.toList
            ExportedBlocks = exportedBlocks |> Seq.distinct |> Seq.toList
        }
    
    /// <summary>
    /// Parse multiple chunks of code in parallel.
    /// </summary>
    /// <param name="chunks">The chunks to parse.</param>
    /// <param name="parser">The parser function to use.</param>
    /// <returns>A list of chunk parse results.</returns>
    let parseChunksParallel (chunks: CodeChunk list) (parser: string -> TarsProgram) =
        chunks
        |> List.map (fun chunk -> parseChunk chunk parser)
    
    /// <summary>
    /// Parse multiple chunks of code sequentially.
    /// </summary>
    /// <param name="chunks">The chunks to parse.</param>
    /// <param name="parser">The parser function to use.</param>
    /// <returns>A list of chunk parse results.</returns>
    let parseChunksSequential (chunks: CodeChunk list) (parser: string -> TarsProgram) =
        chunks
        |> List.map (fun chunk -> parseChunk chunk parser)
    
    /// <summary>
    /// Adjust diagnostics from a chunk to the original file.
    /// </summary>
    /// <param name="chunk">The chunk containing the diagnostics.</param>
    /// <param name="diagnostics">The diagnostics to adjust.</param>
    /// <returns>A list of adjusted diagnostics.</returns>
    let adjustDiagnostics (chunk: CodeChunk) (diagnostics: Diagnostic list) =
        diagnostics
        |> List.map (fun diagnostic ->
            { diagnostic with
                Line = FileChunker.adjustLineNumber chunk diagnostic.Line
                Column = diagnostic.Column // Column doesn't need adjustment
            }
        )
