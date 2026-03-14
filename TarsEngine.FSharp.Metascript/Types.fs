namespace TarsEngine.FSharp.Metascript

open System
open System.Collections.Generic

/// <summary>
/// Represents a metascript block type.
/// </summary>
type MetascriptBlockType =
    | Text
    | Code
    | FSharp
    | CSharp
    | Python
    | JavaScript
    | SQL
    | Markdown
    | HTML
    | CSS
    | JSON
    | XML
    | YAML
    | Command
    | Query
    | Transformation
    | Analysis
    | Reflection
    | Execution
    | Import
    | Export
    | Unknown

/// <summary>
/// Represents a metascript block parameter.
/// </summary>
type MetascriptBlockParameter = {
    /// <summary>
    /// The name of the parameter.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The value of the parameter.
    /// </summary>
    Value: string
}

/// <summary>
/// Represents a metascript block.
/// </summary>
type MetascriptBlock = {
    /// <summary>
    /// The type of the block.
    /// </summary>
    Type: MetascriptBlockType
    
    /// <summary>
    /// The content of the block.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The line number where the block starts.
    /// </summary>
    LineNumber: int
    
    /// <summary>
    /// The column number where the block starts.
    /// </summary>
    ColumnNumber: int
    
    /// <summary>
    /// The parameters of the block.
    /// </summary>
    Parameters: MetascriptBlockParameter list
    
    /// <summary>
    /// The ID of the block.
    /// </summary>
    Id: string
    
    /// <summary>
    /// The parent block ID, if any.
    /// </summary>
    ParentId: string option
    
    /// <summary>
    /// Additional metadata for the block.
    /// </summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Represents a metascript variable.
/// </summary>
type MetascriptVariable = {
    /// <summary>
    /// The name of the variable.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The value of the variable.
    /// </summary>
    Value: obj
    
    /// <summary>
    /// The type of the variable.
    /// </summary>
    Type: Type
    
    /// <summary>
    /// Whether the variable is read-only.
    /// </summary>
    IsReadOnly: bool
    
    /// <summary>
    /// Additional metadata for the variable.
    /// </summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Represents a metascript.
/// </summary>
type Metascript = {
    /// <summary>
    /// The name of the metascript.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The blocks in the metascript.
    /// </summary>
    Blocks: MetascriptBlock list
    
    /// <summary>
    /// The file path of the metascript.
    /// </summary>
    FilePath: string option
    
    /// <summary>
    /// The creation time of the metascript.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The last modification time of the metascript.
    /// </summary>
    LastModificationTime: DateTime option
    
    /// <summary>
    /// The description of the metascript.
    /// </summary>
    Description: string option
    
    /// <summary>
    /// The author of the metascript.
    /// </summary>
    Author: string option
    
    /// <summary>
    /// The version of the metascript.
    /// </summary>
    Version: string option
    
    /// <summary>
    /// The dependencies of the metascript.
    /// </summary>
    Dependencies: string list
    
    /// <summary>
    /// The imports of the metascript.
    /// </summary>
    Imports: string list
    
    /// <summary>
    /// Additional metadata for the metascript.
    /// </summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Represents a metascript context.
/// </summary>
type MetascriptContext = {
    /// <summary>
    /// The variables in the context.
    /// </summary>
    Variables: Map<string, MetascriptVariable>
    
    /// <summary>
    /// The parent context, if any.
    /// </summary>
    Parent: MetascriptContext option
    
    /// <summary>
    /// The current working directory.
    /// </summary>
    WorkingDirectory: string
    
    /// <summary>
    /// The current metascript.
    /// </summary>
    CurrentMetascript: Metascript option
    
    /// <summary>
    /// The current block being executed.
    /// </summary>
    CurrentBlock: MetascriptBlock option
    
    /// <summary>
    /// Additional metadata for the context.
    /// </summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Represents a metascript parser configuration.
/// </summary>
type MetascriptParserConfig = {
    /// <summary>
    /// The block start markers.
    /// </summary>
    BlockStartMarkers: Map<MetascriptBlockType, string>
    
    /// <summary>
    /// The block end markers.
    /// </summary>
    BlockEndMarkers: Map<MetascriptBlockType, string>
    
    /// <summary>
    /// The default block type.
    /// </summary>
    DefaultBlockType: MetascriptBlockType
    
    /// <summary>
    /// Whether to allow nested blocks.
    /// </summary>
    AllowNestedBlocks: bool
    
    /// <summary>
    /// Whether to trim block content.
    /// </summary>
    TrimBlockContent: bool
    
    /// <summary>
    /// Whether to include empty blocks.
    /// </summary>
    IncludeEmptyBlocks: bool
    
    /// <summary>
    /// Additional metadata for the parser configuration.
    /// </summary>
    Metadata: Map<string, string>
}
