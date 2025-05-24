namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of ICodeStructureExtractor for C# language.
/// </summary>
type CSharpStructureExtractor(logger: ILogger<CSharpStructureExtractor>) =
    
    /// <summary>
    /// Gets the language supported by this extractor.
    /// </summary>
    member _.Language = "csharp"
    
    /// <summary>
    /// Extracts code structures from the provided content.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of extracted code structures.</returns>
    member _.ExtractStructures(content: string) =
        try
            logger.LogInformation("Extracting structures from C# code")
            
            // Define regex patterns for different structure types
            let namespacePattern = @"namespace\s+([a-zA-Z0-9_\.]+)\s*\{"
            let classPattern = @"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?::\s*[^{]+)?\s*\{"
            let interfacePattern = @"(public|private|protected|internal)?\s*interface\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?::\s*[^{]+)?\s*\{"
            let enumPattern = @"(public|private|protected|internal)?\s*enum\s+([a-zA-Z0-9_]+)\s*\{"
            let structPattern = @"(public|private|protected|internal)?\s*struct\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?::\s*[^{]+)?\s*\{"
            let methodPattern = @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*(?:where\s+[^{]+)?\s*\{"
            let propertyPattern = @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\{\s*(?:get;)?\s*(?:set;)?\s*\}"
            let fieldPattern = @"(public|private|protected|internal)?\s*(static|readonly|const)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*=?\s*[^;]*;"
            let delegatePattern = @"(public|private|protected|internal)?\s*delegate\s+([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*;"
            let eventPattern = @"(public|private|protected|internal)?\s*event\s+([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*;"
            let usingPattern = @"using\s+(?:static\s+)?([a-zA-Z0-9_\.]+)\s*;"
            let attributePattern = @"\[([a-zA-Z0-9_]+)(?:\(.*\))?\]"
            
            // Extract structures using regex
            let extractStructures pattern structureType =
                Regex.Matches(content, pattern, RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m -> 
                    let startLine = content.Substring(0, m.Index).Split('\n').Length
                    let endLine = startLine + m.Value.Split('\n').Length - 1
                    let modifiers = 
                        m.Groups
                        |> Seq.cast<Group>
                        |> Seq.skip 1 // Skip the full match
                        |> Seq.takeWhile (fun g -> g.Success && g.Value <> "")
                        |> Seq.map (fun g -> g.Value)
                        |> Seq.toList
                    
                    let name = 
                        match structureType with
                        | "namespace" -> m.Groups.[1].Value
                        | "class" | "interface" | "enum" | "struct" -> m.Groups.[3].Value
                        | "method" | "property" | "field" | "delegate" | "event" -> m.Groups.[4].Value
                        | "using" | "attribute" -> m.Groups.[1].Value
                        | _ -> ""
                    
                    let returnType = 
                        match structureType with
                        | "method" | "property" | "field" -> Some m.Groups.[3].Value
                        | "delegate" -> Some m.Groups.[2].Value
                        | _ -> None
                    
                    let parameters = 
                        match structureType with
                        | "method" | "delegate" ->
                            let paramString = m.Groups.[5].Value
                            if String.IsNullOrWhiteSpace(paramString) then
                                []
                            else
                                paramString.Split(',')
                                |> Array.map (fun p -> 
                                    let parts = p.Trim().Split(' ')
                                    if parts.Length >= 2 then
                                        (parts.[0], parts.[1])
                                    else
                                        ("", p.Trim())
                                )
                                |> Array.toList
                        | _ -> []
                    
                    {
                        Name = name
                        StructureType = structureType
                        Parent = None
                        Children = []
                        StartLine = startLine
                        EndLine = endLine
                        Modifiers = modifiers
                        ReturnType = returnType
                        Parameters = parameters
                        Properties = Map.empty
                    }
                )
                |> Seq.toList
            
            // Extract all structures
            let namespaces = extractStructures namespacePattern "namespace"
            let classes = extractStructures classPattern "class"
            let interfaces = extractStructures interfacePattern "interface"
            let enums = extractStructures enumPattern "enum"
            let structs = extractStructures structPattern "struct"
            let methods = extractStructures methodPattern "method"
            let properties = extractStructures propertyPattern "property"
            let fields = extractStructures fieldPattern "field"
            let delegates = extractStructures delegatePattern "delegate"
            let events = extractStructures eventPattern "event"
            let usings = extractStructures usingPattern "using"
            let attributes = extractStructures attributePattern "attribute"
            
            // Combine all structures
            let allStructures = 
                List.concat [
                    namespaces
                    classes
                    interfaces
                    enums
                    structs
                    methods
                    properties
                    fields
                    delegates
                    events
                    usings
                    attributes
                ]
            
            // Build parent-child relationships
            let structureMap = Dictionary<string, CodeStructure>()
            
            // Add structures to map
            for structure in allStructures do
                let key = $"{structure.StructureType}:{structure.Name}"
                if not (structureMap.ContainsKey(key)) then
                    structureMap.Add(key, structure)
            
            // Build parent-child relationships
            let structures = 
                allStructures
                |> List.map (fun structure ->
                    // Find parent
                    let parent = 
                        allStructures
                        |> List.filter (fun s -> 
                            s.StartLine < structure.StartLine && 
                            s.EndLine > structure.EndLine && 
                            s.StructureType <> "using" && 
                            s.StructureType <> "attribute"
                        )
                        |> List.sortByDescending (fun s -> s.StartLine)
                        |> List.tryHead
                    
                    // Find children
                    let children = 
                        allStructures
                        |> List.filter (fun s -> 
                            s.StartLine > structure.StartLine && 
                            s.EndLine < structure.EndLine && 
                            s.StructureType <> "using" && 
                            s.StructureType <> "attribute"
                        )
                        |> List.filter (fun s ->
                            // Ensure the child is a direct child
                            not (allStructures |> List.exists (fun p -> 
                                p.StartLine < s.StartLine && 
                                p.EndLine > s.EndLine && 
                                p.StartLine > structure.StartLine && 
                                p.EndLine < structure.EndLine
                            ))
                        )
                    
                    // Update structure with parent and children
                    { structure with 
                        Parent = parent
                        Children = children
                    }
                )
            
            structures
        with
        | ex ->
            logger.LogError(ex, "Error extracting structures from C# code")
            []
    
    /// <summary>
    /// Extracts code structures from a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of extracted code structures.</returns>
    member this.ExtractStructuresFromFile(filePath: string) =
        try
            logger.LogInformation("Extracting structures from C# file: {FilePath}", filePath)
            
            // Read the file content
            let content = File.ReadAllText(filePath)
            
            // Extract structures from the content
            this.ExtractStructures(content)
        with
        | ex ->
            logger.LogError(ex, "Error extracting structures from C# file: {FilePath}", filePath)
            []
    
    /// <summary>
    /// Gets a structure by name.
    /// </summary>
    /// <param name="structures">The list of structures to search.</param>
    /// <param name="name">The name of the structure to find.</param>
    /// <returns>The found structure, if any.</returns>
    member _.GetStructureByName(structures: CodeStructure list, name: string) =
        structures |> List.tryFind (fun s -> s.Name = name)
    
    /// <summary>
    /// Gets structures by type.
    /// </summary>
    /// <param name="structures">The list of structures to search.</param>
    /// <param name="structureType">The type of structures to find.</param>
    /// <returns>The list of found structures.</returns>
    member _.GetStructuresByType(structures: CodeStructure list, structureType: string) =
        structures |> List.filter (fun s -> s.StructureType = structureType)
    
    interface ICodeStructureExtractor with
        member this.Language = this.Language
        member this.ExtractStructures(content) = this.ExtractStructures(content)
        member this.ExtractStructuresFromFile(filePath) = this.ExtractStructuresFromFile(filePath)
        member this.GetStructureByName(structures, name) = this.GetStructureByName(structures, name)
        member this.GetStructuresByType(structures, structureType) = this.GetStructuresByType(structures, structureType)
