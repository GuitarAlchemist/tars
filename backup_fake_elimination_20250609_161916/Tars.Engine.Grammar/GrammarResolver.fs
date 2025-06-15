namespace Tars.Engine.Grammar

open System
open System.IO
open System.Text.Json
open System.Collections.Generic

/// Grammar resolution and management
module GrammarResolver =
    
    let private grammarsDirectory = ".tars/grammars"
    let private grammarIndexFile = ".tars/grammars/grammar_index.json"
    
    /// Grammar index entry
    type GrammarIndexEntry = {
        Id: string
        File: string
        Origin: string
        Version: string option
        Hash: string option
        LastModified: DateTime option
    }
    
    /// Ensure grammars directory exists
    let ensureGrammarsDirectory () =
        if not (Directory.Exists(grammarsDirectory)) then
            Directory.CreateDirectory(grammarsDirectory) |> ignore
    
    /// Get grammar file path
    let getGrammarFilePath name =
        Path.Combine(grammarsDirectory, $"{name}.tars")
    
    /// Try to resolve grammar by name from external files
    let tryResolveExternal name =
        ensureGrammarsDirectory()
        let filePath = getGrammarFilePath name
        let file = FileInfo(filePath)
        if file.Exists then
            Some (GrammarSource.External file)
        else
            None
    
    /// Try to resolve grammar from inline definitions in current context
    let tryResolveInline name (inlineGrammars: Map<string, string>) =
        Map.tryFind name inlineGrammars
        |> Option.map (fun content -> GrammarSource.Inline (name, content))
    
    /// Resolve grammar with fallback hierarchy
    let resolveGrammar name (inlineGrammars: Map<string, string>) =
        match tryResolveExternal name with
        | Some external -> Some external
        | None -> tryResolveInline name inlineGrammars
    
    /// Load grammar index from file
    let loadGrammarIndex () =
        try
            if File.Exists(grammarIndexFile) then
                let json = File.ReadAllText(grammarIndexFile)
                JsonSerializer.Deserialize<GrammarIndexEntry[]>(json)
                |> Array.toList
            else
                []
        with
        | ex ->
            printfn $"Warning: Failed to load grammar index: {ex.Message}"
            []
    
    /// Save grammar index to file
    let saveGrammarIndex (entries: GrammarIndexEntry list) =
        try
            ensureGrammarsDirectory()
            let json = JsonSerializer.Serialize(entries, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(grammarIndexFile, json)
        with
        | ex ->
            printfn $"Warning: Failed to save grammar index: {ex.Message}"
    
    /// Update grammar index with new entry
    let updateGrammarIndex entry =
        let index = loadGrammarIndex()
        let updatedIndex = 
            index
            |> List.filter (fun e -> e.Id <> entry.Id)
            |> List.append [entry]
        saveGrammarIndex updatedIndex
    
    /// Remove entry from grammar index
    let removeFromGrammarIndex grammarId =
        let index = loadGrammarIndex()
        let updatedIndex = index |> List.filter (fun e -> e.Id <> grammarId)
        saveGrammarIndex updatedIndex
    
    /// List all available grammars
    let listGrammars () =
        ensureGrammarsDirectory()
        let externalGrammars = 
            Directory.GetFiles(grammarsDirectory, "*.tars")
            |> Array.map (fun path -> 
                let file = FileInfo(path)
                let name = Path.GetFileNameWithoutExtension(file.Name)
                (name, GrammarSource.External file))
            |> Array.toList
        
        externalGrammars
    
    /// Check if grammar exists
    let grammarExists name =
        let filePath = getGrammarFilePath name
        File.Exists(filePath)
    
    /// Save grammar to external file
    let saveGrammar (grammar: Grammar) =
        try
            ensureGrammarsDirectory()
            let filePath = getGrammarFilePath grammar.Metadata.Name

            // Create TARS grammar file content
            let content =
                sprintf """meta {
  name: "%s"
  version: "%s"
  source: "%s"
  language: "%s"
  created: "%s"
  description: "%s"
}

grammar {
  LANG("%s") {
%s
  }
}"""
                    grammar.Metadata.Name
                    (grammar.Metadata.Version |> Option.defaultValue "v1.0")
                    (grammar.Metadata.Source |> Option.defaultValue "manual")
                    grammar.Metadata.Language
                    (grammar.Metadata.Created |> Option.map (fun d -> d.ToString("yyyy-MM-dd HH:mm:ss")) |> Option.defaultValue "")
                    (grammar.Metadata.Description |> Option.defaultValue "")
                    grammar.Metadata.Language
                    (grammar.Content.Split('\n') |> Array.map (fun line -> "    " + line) |> String.concat "\n")

            File.WriteAllText(filePath, content)

            // Update index
            let entry = {
                Id = grammar.Metadata.Name
                File = Path.GetFileName(filePath)
                Origin = grammar.Metadata.Source |> Option.defaultValue "manual"
                Version = grammar.Metadata.Version
                Hash = grammar.Metadata.Hash
                LastModified = Some DateTime.Now
            }
            updateGrammarIndex entry

            printfn "✅ Grammar '%s' saved to %s" grammar.Metadata.Name filePath
            true
        with
        | ex ->
            printfn "❌ Failed to save grammar '%s': %s" grammar.Metadata.Name ex.Message
            false
    
    /// Load grammar from external file
    let loadGrammar name =
        match tryResolveExternal name with
        | Some (GrammarSource.External file) ->
            try
                let grammar = Grammar.createExternal file
                Some grammar
            with
            | ex ->
                printfn "❌ Failed to load grammar '%s': %s" name ex.Message
                None
        | _ -> None
    
    /// Extract inline grammar to external file
    let extractInlineGrammar name content =
        let grammar = Grammar.createInline name content
        if saveGrammar grammar then
            printfn "✅ Extracted inline grammar '%s' to external file" name
            true
        else
            false
    
    /// Inline external grammar (return content for embedding)
    let inlineGrammar name =
        match loadGrammar name with
        | Some grammar ->
            Some grammar.Content
        | None ->
            printfn "❌ Grammar '%s' not found" name
            None
    
    /// Detect duplicate inline grammars and suggest extraction
    let detectDuplicateInlineGrammars (inlineGrammars: Map<string, string>) =
        let duplicates = 
            inlineGrammars
            |> Map.toList
            |> List.groupBy snd
            |> List.filter (fun (_, grammars) -> List.length grammars > 1)
            |> List.map (fun (content, grammars) -> 
                let names = grammars |> List.map fst
                (content, names))
        
        if not (List.isEmpty duplicates) then
            printfn "💡 Duplicate inline grammars detected:"
            duplicates |> List.iter (fun (_, names) ->
                printfn "   Grammar appears in: %s" (String.concat ", " names))
            printfn "   Consider extracting to external files using: tarscli grammar extract-from-meta"
    
    /// Get grammar statistics
    let getGrammarStats name =
        match loadGrammar name with
        | Some grammar ->
            let size = GrammarSource.getSize grammar.Source
            let lines = grammar.Content.Split('\n').Length
            let lastModified = GrammarSource.getLastModified grammar.Source
            Some {|
                Name = name
                Size = size
                Lines = lines
                LastModified = lastModified
                Hash = grammar.Metadata.Hash
                Version = grammar.Metadata.Version
            |}
        | None -> None
    
    /// Validate grammar syntax (basic validation)
    let validateGrammar grammar =
        let errors = ResizeArray<string>()
        
        // Basic EBNF validation
        if String.IsNullOrWhiteSpace(grammar.Content) then
            errors.Add("Grammar content is empty")
        
        if not (grammar.Content.Contains("=")) then
            errors.Add("Grammar appears to be missing production rules (no '=' found)")
        
        // Check for balanced braces/brackets
        let openBraces = grammar.Content |> Seq.filter ((=) '{') |> Seq.length
        let closeBraces = grammar.Content |> Seq.filter ((=) '}') |> Seq.length
        if openBraces <> closeBraces then
            errors.Add(sprintf "Unbalanced braces: %d open, %d close" openBraces closeBraces)

        let openBrackets = grammar.Content |> Seq.filter ((=) '[') |> Seq.length
        let closeBrackets = grammar.Content |> Seq.filter ((=) ']') |> Seq.length
        if openBrackets <> closeBrackets then
            errors.Add(sprintf "Unbalanced brackets: %d open, %d close" openBrackets closeBrackets)
        
        if errors.Count = 0 then
            Ok grammar
        else
            Error (errors |> Seq.toList)
    
    /// Generate grammar from examples (placeholder for future ML implementation)
    let generateGrammarFromExamples name examples =
        // TODO: Implement ML-based grammar generation
        let placeholderGrammar = sprintf """// Generated grammar for %s
start = expression ;
expression = term , { ( "+" | "-" ) , term } ;
term = factor , { ( "*" | "/" ) , factor } ;
factor = number | "(" , expression , ")" ;
number = digit , { digit } ;
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;""" name
        
        let grammar = Grammar.createInline name placeholderGrammar
        grammar
