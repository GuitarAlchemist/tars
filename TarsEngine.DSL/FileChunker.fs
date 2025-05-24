namespace TarsEngine.DSL

open System
open System.IO
open System.Text

/// <summary>
/// Represents a chunk of code from a file.
/// </summary>
type CodeChunk = {
    /// <summary>
    /// The content of the chunk.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The start line number of the chunk in the original file (1-based).
    /// </summary>
    StartLine: int
    
    /// <summary>
    /// The end line number of the chunk in the original file (1-based).
    /// </summary>
    EndLine: int
    
    /// <summary>
    /// The start position of the chunk in the original file (0-based).
    /// </summary>
    StartPosition: int
    
    /// <summary>
    /// The end position of the chunk in the original file (0-based).
    /// </summary>
    EndPosition: int
    
    /// <summary>
    /// The index of the chunk in the sequence of chunks.
    /// </summary>
    ChunkIndex: int
}

/// <summary>
/// Module for chunking large files into smaller pieces for incremental parsing.
/// </summary>
module FileChunker =
    /// <summary>
    /// Default chunk size in lines.
    /// </summary>
    let defaultChunkSize = 1000
    
    /// <summary>
    /// Maximum chunk size in lines.
    /// </summary>
    let maxChunkSize = 10000
    
    /// <summary>
    /// Minimum chunk size in lines.
    /// </summary>
    let minChunkSize = 100
    
    /// <summary>
    /// Check if a line is a potential chunk boundary.
    /// A chunk boundary is a line that starts a new block or is empty.
    /// </summary>
    /// <param name="line">The line to check.</param>
    /// <returns>True if the line is a potential chunk boundary, false otherwise.</returns>
    let isPotentialChunkBoundary (line: string) =
        let trimmedLine = line.Trim()
        
        // Empty lines are potential chunk boundaries
        if String.IsNullOrWhiteSpace(trimmedLine) then
            true
        // Lines that start a new block are potential chunk boundaries
        elif trimmedLine.EndsWith("{") then
            true
        // Lines that end a block are potential chunk boundaries
        elif trimmedLine = "}" then
            true
        // Comment lines are potential chunk boundaries
        elif trimmedLine.StartsWith("//") || trimmedLine.StartsWith("/*") || trimmedLine.EndsWith("*/") then
            true
        else
            false
    
    /// <summary>
    /// Split a file into chunks.
    /// </summary>
    /// <param name="filePath">The path to the file to chunk.</param>
    /// <param name="chunkSize">The target chunk size in lines.</param>
    /// <returns>A list of code chunks.</returns>
    let chunkFile (filePath: string) (chunkSize: int) =
        // Validate chunk size
        let validatedChunkSize = 
            if chunkSize < minChunkSize then minChunkSize
            elif chunkSize > maxChunkSize then maxChunkSize
            else chunkSize
        
        // Read the file
        let fileContent = File.ReadAllText(filePath)
        let lines = fileContent.Split([|'\n'|], StringSplitOptions.None)
        
        // Calculate line start positions
        let lineStartPositions = 
            lines 
            |> Array.scan (fun pos line -> pos + line.Length + 1) 0 
            |> Array.take lines.Length
        
        // Find chunk boundaries
        let chunks = ResizeArray<CodeChunk>()
        let mutable chunkStartLine = 0
        let mutable chunkStartPosition = 0
        let mutable chunkIndex = 0
        
        for i in 0 .. lines.Length - 1 do
            let lineNumber = i + 1 // Convert to 1-based line number
            let linePosition = lineStartPositions.[i]
            
            // Check if we should end the current chunk
            if i > 0 && 
               (i - chunkStartLine >= validatedChunkSize) && 
               isPotentialChunkBoundary lines.[i] then
                
                // Create a chunk from chunkStartLine to i-1
                let chunkContent = 
                    lines.[chunkStartLine..i-1] 
                    |> String.concat "\n"
                
                let chunk = {
                    Content = chunkContent
                    StartLine = chunkStartLine + 1 // Convert to 1-based line number
                    EndLine = i // Convert to 1-based line number
                    StartPosition = chunkStartPosition
                    EndPosition = linePosition - 1
                    ChunkIndex = chunkIndex
                }
                
                chunks.Add(chunk)
                
                // Start a new chunk
                chunkStartLine <- i
                chunkStartPosition <- linePosition
                chunkIndex <- chunkIndex + 1
        
        // Add the last chunk if needed
        if chunkStartLine < lines.Length then
            let chunkContent = 
                lines.[chunkStartLine..] 
                |> String.concat "\n"
            
            let chunk = {
                Content = chunkContent
                StartLine = chunkStartLine + 1 // Convert to 1-based line number
                EndLine = lines.Length // Convert to 1-based line number
                StartPosition = chunkStartPosition
                EndPosition = fileContent.Length
                ChunkIndex = chunkIndex
            }
            
            chunks.Add(chunk)
        
        chunks |> Seq.toList
    
    /// <summary>
    /// Split a string into chunks.
    /// </summary>
    /// <param name="content">The content to chunk.</param>
    /// <param name="chunkSize">The target chunk size in lines.</param>
    /// <returns>A list of code chunks.</returns>
    let chunkString (content: string) (chunkSize: int) =
        // Validate chunk size
        let validatedChunkSize = 
            if chunkSize < minChunkSize then minChunkSize
            elif chunkSize > maxChunkSize then maxChunkSize
            else chunkSize
        
        // Split the content into lines
        let lines = content.Split([|'\n'|], StringSplitOptions.None)
        
        // Calculate line start positions
        let lineStartPositions = 
            lines 
            |> Array.scan (fun pos line -> pos + line.Length + 1) 0 
            |> Array.take lines.Length
        
        // Find chunk boundaries
        let chunks = ResizeArray<CodeChunk>()
        let mutable chunkStartLine = 0
        let mutable chunkStartPosition = 0
        let mutable chunkIndex = 0
        
        for i in 0 .. lines.Length - 1 do
            let lineNumber = i + 1 // Convert to 1-based line number
            let linePosition = lineStartPositions.[i]
            
            // Check if we should end the current chunk
            if i > 0 && 
               (i - chunkStartLine >= validatedChunkSize) && 
               isPotentialChunkBoundary lines.[i] then
                
                // Create a chunk from chunkStartLine to i-1
                let chunkContent = 
                    lines.[chunkStartLine..i-1] 
                    |> String.concat "\n"
                
                let chunk = {
                    Content = chunkContent
                    StartLine = chunkStartLine + 1 // Convert to 1-based line number
                    EndLine = i // Convert to 1-based line number
                    StartPosition = chunkStartPosition
                    EndPosition = linePosition - 1
                    ChunkIndex = chunkIndex
                }
                
                chunks.Add(chunk)
                
                // Start a new chunk
                chunkStartLine <- i
                chunkStartPosition <- linePosition
                chunkIndex <- chunkIndex + 1
        
        // Add the last chunk if needed
        if chunkStartLine < lines.Length then
            let chunkContent = 
                lines.[chunkStartLine..] 
                |> String.concat "\n"
            
            let chunk = {
                Content = chunkContent
                StartLine = chunkStartLine + 1 // Convert to 1-based line number
                EndLine = lines.Length // Convert to 1-based line number
                StartPosition = chunkStartPosition
                EndPosition = content.Length
                ChunkIndex = chunkIndex
            }
            
            chunks.Add(chunk)
        
        chunks |> Seq.toList
    
    /// <summary>
    /// Adjust a line number from a chunk to the original file.
    /// </summary>
    /// <param name="chunk">The chunk containing the line.</param>
    /// <param name="chunkLineNumber">The line number in the chunk (1-based).</param>
    /// <returns>The line number in the original file (1-based).</returns>
    let adjustLineNumber (chunk: CodeChunk) (chunkLineNumber: int) =
        chunk.StartLine + chunkLineNumber - 1
    
    /// <summary>
    /// Adjust a position from a chunk to the original file.
    /// </summary>
    /// <param name="chunk">The chunk containing the position.</param>
    /// <param name="chunkPosition">The position in the chunk (0-based).</param>
    /// <returns>The position in the original file (0-based).</returns>
    let adjustPosition (chunk: CodeChunk) (chunkPosition: int) =
        chunk.StartPosition + chunkPosition
