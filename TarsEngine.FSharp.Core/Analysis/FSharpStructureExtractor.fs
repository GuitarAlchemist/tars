namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of ICodeStructureExtractor for F# language.
/// </summary>
type FSharpStructureExtractor(logger: ILogger<FSharpStructureExtractor>) =
    
    /// <summary>
    /// Gets the language supported by this extractor.
    /// </summary>
    member _.Language = "fsharp"
    
    /// <summary>
    /// Extracts code structures from the provided content.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of extracted code structures.</returns>
    member _.ExtractStructures(content: string) =
        try
            logger.LogInformation("Extracting structures from F# code")
            
            // Define regex patterns for different structure types
            let namespacePattern = @"namespace\s+([a-zA-Z0-9_\.]+)"
            let modulePattern = @"module\s+(?:rec\s+)?([a-zA-Z0-9_\.]+)"
            let typePattern = @"type\s+(?:private\s+)?([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?:=\s*[^{]+)?\s*(?:\{|=|\()"
            let recordPattern = @"type\s+(?:private\s+)?([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*=\s*\{[^}]*\}"
            let unionPattern = @"type\s+(?:private\s+)?([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*=(?:\s*\|)?\s*([^=]+)(?:\s*\|[^=]+)*"
            let functionPattern = @"let\s+(?:rec\s+)?(?:private\s+)?([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?:\([^)]*\))?\s*(?::\s*[^=]+)?\s*="
            let valuePattern = @"let\s+(?:mutable\s+)?(?:private\s+)?([a-zA-Z0-9_]+)\s*(?::\s*[^=]+)?\s*="
            let memberPattern = @"member\s+(?:private\s+)?(?:this|self|_)\.([a-zA-Z0-9_]+)(?:<[^>]+>)?\s*(?:\([^)]*\))?\s*(?::\s*[^=]+)?\s*="
            let propertyPattern = @"member\s+(?:private\s+)?(?:this|self|_)\.([a-zA-Z0-9_]+)\s*(?::\s*[^=]+)?\s*with\s+(?:get|set)"
            let interfacePattern = @"interface\s+([a-zA-Z0-9_\.]+)(?:<[^>]+>)?"
            let openPattern = @"open\s+([a-zA-Z0-9_\.]+)"
            let attributePattern = @"\[<([a-zA-Z0-9_]+)(?:\(.*\))?\s*>\]"
            
            // Extract structures using regex
            let extractStructures pattern structureType =
                Regex.Matches(content, pattern, RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m -> 
                    let startLine = content.Substring(0, m.Index).Split('\n').Length
                    let endLine = startLine + m.Value.Split('\n').Length - 1
                    let modifiers = 
                        if m.Value.Contains("private") then ["private"]
                        elif m.Value.Contains("internal") then ["internal"]
                        else ["public"]
                    
                    let name = 
                        match structureType with
                        | "namespace" | "module" | "type" | "record" | "union" | "function" | "value" | "member" | "property" | "interface" | "open" | "attribute" ->
                            m.Groups.[1].Value
                        | _ -> ""
                    
                    let returnType = 
                        let returnTypeMatch = Regex.Match(m.Value, @":\s*([^=]+)")
                        if returnTypeMatch.Success then
                            Some (returnTypeMatch.Groups.[1].Value.Trim())
                        else
                            None
                    
                    let parameters = 
                        match structureType with
                        | "function" | "member" ->
                            let paramMatch = Regex.Match(m.Value, @"\(([^)]*)\)")
                            if paramMatch.Success then
                                let paramString = paramMatch.Groups.[1].Value
                                if String.IsNullOrWhiteSpace(paramString) then
                                    []
                                else
                                    paramString.Split(',')
                                    |> Array.map (fun p -> 
                                        let parts = p.Trim().Split(':')
                                        if parts.Length >= 2 then
                                            (parts.[0].Trim(), parts.[1].Trim())
                                        else
                                            ("", p.Trim())
                                    )
                                    |> Array.toList
                            else
                                []
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
            let modules = extractStructures modulePattern "module"
            let types = extractStructures typePattern "type"
            let records = extractStructures recordPattern "record"
            let unions = extractStructures unionPattern "union"
            let functions = extractStructures functionPattern "function"
            let values = extractStructures valuePattern "value"
            let members = extractStructures memberPattern "member"
            let properties = extractStructures propertyPattern "property"
            let interfaces = extractStructures interfacePattern "interface"
            let opens = extractStructures openPattern "open"
            let attributes = extractStructures attributePattern "attribute"
            
            // Combine all structures
            let allStructures = 
                List.concat [
                    namespaces
                    modules
                    types
                    records
                    unions
                    functions
                    values
                    members
                    properties
                    interfaces
                    opens
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
                            s.StructureType <> "open" && 
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
                            s.StructureType <> "open" && 
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
            logger.LogError(ex, "Error extracting structures from F# code")
            []
    
    /// <summary>
    /// Extracts code structures from a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of extracted code structures.</returns>
    member this.ExtractStructuresFromFile(filePath: string) =
        try
            logger.LogInformation("Extracting structures from F# file: {FilePath}", filePath)
            
            // Read the file content
            let content = File.ReadAllText(filePath)
            
            // Extract structures from the content
            this.ExtractStructures(content)
        with
        | ex ->
            logger.LogError(ex, "Error extracting structures from F# file: {FilePath}", filePath)
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
