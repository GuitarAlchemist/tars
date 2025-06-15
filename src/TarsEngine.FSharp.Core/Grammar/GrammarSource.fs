namespace Tars.Engine.Grammar

open System
open System.IO
open System.Security.Cryptography
open System.Text

/// Represents the source of a grammar definition
type GrammarSource =
    | Inline of name: string * content: string
    | External of file: FileInfo
    | EmbeddedRFC of rfcId: string * ruleName: string

/// Metadata for a grammar definition
type GrammarMetadata = {
    Name: string
    Version: string option
    Source: string option  // "rfc", "generated", "manual"
    Language: string       // "EBNF", "BNF", etc.
    Created: DateTime option
    LastModified: DateTime option
    Hash: string option
    Description: string option
    Tags: string list
}

/// A complete grammar definition with metadata
type Grammar = {
    Metadata: GrammarMetadata
    Source: GrammarSource
    Content: string
}

/// Language block for multi-language support
type LanguageBlock = {
    Language: string
    Code: string
    Metadata: Map<string, string>
    EntryPoint: string option
    Dependencies: string list
}

/// RFC reference for standards-based development
type RFCReference = {
    RFCId: string
    Title: string option
    Url: string option
    ExtractRules: string list
    VerifyCompatibilityWith: string option
    UseIn: string option
}

/// Extended metascript block types
type ExtendedBlockType =
    | FSharpBlock of string
    | CSharpBlock of string
    | LanguageBlock of LanguageBlock
    | GrammarBlock of Grammar
    | RFCBlock of RFCReference

module GrammarSource =
    
    /// Get the name of a grammar source
    let getName = function
        | Inline (name, _) -> name
        | External file -> Path.GetFileNameWithoutExtension(file.Name)
        | EmbeddedRFC (rfcId, ruleName) -> sprintf "%s_%s" rfcId ruleName
    
    /// Get the content of a grammar source
    let getContent = function
        | Inline (_, content) -> content
        | External file -> 
            if file.Exists then File.ReadAllText(file.FullName)
            else failwith $"Grammar file not found: {file.FullName}"
        | EmbeddedRFC (rfcId, ruleName) -> 
            // TODO: Implement RFC rule extraction
            failwith $"RFC extraction not yet implemented for {rfcId}:{ruleName}"
    
    /// Check if a grammar source exists
    let exists = function
        | Inline _ -> true
        | External file -> file.Exists
        | EmbeddedRFC _ -> false // TODO: Implement RFC availability check
    
    /// Get the last modified time of a grammar source
    let getLastModified = function
        | Inline _ -> DateTime.Now
        | External file when file.Exists -> file.LastWriteTime
        | External _ -> DateTime.MinValue
        | EmbeddedRFC _ -> DateTime.MinValue
    
    /// Get the size of a grammar source in bytes
    let getSize = function
        | Inline (_, content) -> int64 (Encoding.UTF8.GetByteCount(content))
        | External file when file.Exists -> file.Length
        | External _ -> 0L
        | EmbeddedRFC _ -> 0L
    
    /// Compute SHA256 hash of grammar content
    let computeHash source =
        let content = getContent source
        use sha256 = SHA256.Create()
        let hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(content))
        Convert.ToHexString(hashBytes).ToLowerInvariant()

module GrammarMetadata =
    
    /// Create default metadata for a grammar
    let createDefault name =
        {
            Name = name
            Version = Some "v1.0"
            Source = Some "manual"
            Language = "EBNF"
            Created = Some DateTime.Now
            LastModified = Some DateTime.Now
            Hash = None
            Description = None
            Tags = []
        }
    
    /// Update metadata with computed values
    let updateComputed metadata grammarSource =
        let hash = GrammarSource.computeHash grammarSource
        let lastModified = GrammarSource.getLastModified grammarSource
        { metadata with 
            Hash = Some hash
            LastModified = Some lastModified }

module Grammar =
    
    /// Create a new grammar from source
    let create name source =
        let metadata = GrammarMetadata.createDefault name
        let content = GrammarSource.getContent source
        let updatedMetadata = GrammarMetadata.updateComputed metadata source
        {
            Metadata = updatedMetadata
            Source = source
            Content = content
        }
    
    /// Create an inline grammar
    let createInline name content =
        let source = Inline (name, content)
        create name source
    
    /// Create an external grammar from file
    let createExternal (file: System.IO.FileInfo) =
        let name = Path.GetFileNameWithoutExtension(file.Name)
        let source = External file
        create name source
    
    /// Check if grammar needs updating (based on file modification time)
    let needsUpdate grammar =
        match grammar.Source with
        | External file when file.Exists ->
            match grammar.Metadata.LastModified with
            | Some lastMod -> file.LastWriteTime > lastMod
            | None -> true
        | _ -> false
    
    /// Update grammar content and metadata
    let update grammar =
        if needsUpdate grammar then
            let newContent = GrammarSource.getContent grammar.Source
            let newMetadata = GrammarMetadata.updateComputed grammar.Metadata grammar.Source
            { grammar with 
                Content = newContent
                Metadata = newMetadata }
        else
            grammar

module LanguageBlock =
    
    /// Create a language block
    let create (language: string) (code: string) =
        {
            Language = language.ToUpperInvariant()
            Code = code
            Metadata = Map.empty
            EntryPoint = None
            Dependencies = []
        }
    
    /// Add metadata to a language block
    let withMetadata key value block =
        { block with Metadata = Map.add key value block.Metadata }
    
    /// Set entry point for a language block
    let withEntryPoint entryPoint block =
        { block with EntryPoint = Some entryPoint }
    
    /// Add dependencies to a language block
    let withDependencies deps block =
        { block with Dependencies = deps }
    
    /// Check if language is supported
    let isSupported (language: string) =
        match language.ToUpperInvariant() with
        | "FSHARP" | "CSHARP" | "PYTHON" | "RUST" | "JAVASCRIPT"
        | "TYPESCRIPT" | "POWERSHELL" | "BASH" | "SQL" | "WOLFRAM" | "JULIA" -> true
        | _ -> false

module RFCReference =
    
    /// Create an RFC reference
    let create rfcId =
        {
            RFCId = rfcId
            Title = None
            Url = None
            ExtractRules = []
            VerifyCompatibilityWith = None
            UseIn = None
        }
    
    /// Set RFC title
    let withTitle title rfc =
        { rfc with Title = Some title }
    
    /// Set RFC URL
    let withUrl url rfc =
        { rfc with Url = Some url }
    
    /// Add rules to extract
    let withExtractRules rules rfc =
        { rfc with ExtractRules = rules }
    
    /// Set compatibility verification target
    let withCompatibilityCheck target rfc =
        { rfc with VerifyCompatibilityWith = Some target }
    
    /// Set usage context
    let withUsage usage rfc =
        { rfc with UseIn = Some usage }
    
    /// Get standard RFC URL
    let getStandardUrl (rfcId: string) =
        $"https://datatracker.ietf.org/doc/html/{rfcId.ToLowerInvariant()}"
