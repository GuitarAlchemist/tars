namespace TarsEngine.FSharp.Core.Metascript

open System
open TarsEngine.FSharp.Core.Types

/// Metascript types
module Types =
    
    /// Metascript metadata
    type MetascriptMeta = {
        Name: string
        Version: string
        Description: string
        Author: string
        Created: string
        Tags: string list
        ExecutionMode: string
        OutputDirectory: string
        TraceEnabled: bool
    }
    
    /// Metascript block type
    type BlockType =
        | Meta
        | Reasoning
        | FSharp
        | Lang of string
    
    /// Metascript block
    type MetascriptBlock = {
        Type: BlockType
        Content: string
        LineNumber: int
    }
    
    /// Parsed metascript
    type ParsedMetascript = {
        Meta: MetascriptMeta option
        Blocks: MetascriptBlock list
        FilePath: string
    }
