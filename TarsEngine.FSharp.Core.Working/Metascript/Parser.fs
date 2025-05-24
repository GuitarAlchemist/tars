namespace TarsEngine.FSharp.Core.Working.Metascript

open System
open System.IO
open System.Text.RegularExpressions
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Represents a metascript block.
/// </summary>
type MetascriptBlock =
    | ConfigBlock of Map<string, obj>
    | FSharpBlock of string
    | TextBlock of string
    | CommandBlock of string

/// <summary>
/// Represents a parsed metascript.
/// </summary>
type Metascript = {
    Id: Id
    Name: string
    FilePath: string option
    Blocks: MetascriptBlock list
    Metadata: Metadata
}

/// <summary>
/// Enhanced metascript parser.
/// </summary>
type MetascriptParser() =
    
    /// <summary>
    /// Parses a metascript from text.
    /// </summary>
    member this.Parse(text: string, filePath: string option) =
        let id = Guid.NewGuid().ToString()
        let name = 
            match filePath with
            | Some path -> Path.GetFileNameWithoutExtension(path)
            | None -> "unnamed"
        
        // Parse blocks from the text
        let blocks = this.parseBlocks text
        
        {
            Id = id
            Name = name
            FilePath = filePath
            Blocks = blocks
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Parses blocks from metascript text.
    /// </summary>
    member private this.parseBlocks(text: string) =
        let mutable blocks = []
        let mutable remainingText = text
        
        // Define patterns for different block types
        let configPattern = @"CONFIG\s*\{([^}]*)\}"
        let fsharpPattern = @"FSHARP\s*\{([^}]*)\}"
        let commandPattern = @"COMMAND\s*\{([^}]*)\}"
        
        // Parse CONFIG blocks
        let configMatches = Regex.Matches(remainingText, configPattern, RegexOptions.Singleline)
        for m in configMatches do
            let configText = m.Groups.[1].Value.Trim()
            let configMap = this.parseConfigText configText
            blocks <- ConfigBlock configMap :: blocks
            remainingText <- remainingText.Replace(m.Value, "")
        
        // Parse FSHARP blocks
        let fsharpMatches = Regex.Matches(remainingText, fsharpPattern, RegexOptions.Singleline)
        for m in fsharpMatches do
            let fsharpCode = m.Groups.[1].Value.Trim()
            blocks <- FSharpBlock fsharpCode :: blocks
            remainingText <- remainingText.Replace(m.Value, "")
        
        // Parse COMMAND blocks
        let commandMatches = Regex.Matches(remainingText, commandPattern, RegexOptions.Singleline)
        for m in commandMatches do
            let commandText = m.Groups.[1].Value.Trim()
            blocks <- CommandBlock commandText :: blocks
            remainingText <- remainingText.Replace(m.Value, "")
        
        // Any remaining text is treated as a text block
        let remainingText = remainingText.Trim()
        if not (String.IsNullOrEmpty(remainingText)) then
            blocks <- TextBlock remainingText :: blocks
        
        List.rev blocks
    
    /// <summary>
    /// Parses configuration text into a map.
    /// </summary>
    member private this.parseConfigText(text: string) =
        let lines = text.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        let mutable configMap = Map.empty
        
        for line in lines do
            let trimmedLine = line.Trim()
            if not (String.IsNullOrEmpty(trimmedLine)) && not (trimmedLine.StartsWith("//")) then
                let parts = trimmedLine.Split([|':'|], 2)
                if parts.Length = 2 then
                    let key = parts.[0].Trim()
                    let value = parts.[1].Trim().Trim([|'"'; '\''|])
                    configMap <- configMap.Add(key, value :> obj)
        
        configMap

/// <summary>
/// Module functions for parsing metascripts.
/// </summary>
module MetascriptParser =
    
    /// <summary>
    /// Parses a metascript from a file.
    /// </summary>
    let parseFile filePath =
        let parser = MetascriptParser()
        let text = File.ReadAllText(filePath)
        parser.Parse(text, Some filePath)
    
    /// <summary>
    /// Parses a metascript from text.
    /// </summary>
    let parseText text =
        let parser = MetascriptParser()
        parser.Parse(text, None)
