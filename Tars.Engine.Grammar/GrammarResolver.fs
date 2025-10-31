namespace Tars.Engine.Grammar

open System
open System.IO
open System.Text.Json
open System.Collections.Generic
open System.Text
open System.Text.RegularExpressions

/// Grammar resolution and management
module GrammarResolver =
    
    let private sanitizeIdentifier (input: string) =
        let sanitized = Regex.Replace(input.ToLowerInvariant(), "[^a-z0-9]+", "_").Trim([|'_'|])
        if String.IsNullOrWhiteSpace(sanitized) then "example" else sanitized

    let private tokenizeExample (example: string) =
        Regex.Matches(example, @"[A-Za-z0-9_]+|\S")
        |> Seq.cast<System.Text.RegularExpressions.Match>
        |> Seq.map (fun m -> m.Value)
        |> Seq.toList

    let private encodeToken (token: string) =
        let escaped = token.Replace("\"", "\\\"")
        $"\"{escaped}\""
    
    let private grammarsDirectory = ".tars/evolution/grammars/base"
    let private grammarIndexFile = ".tars/evolution/grammars/base/grammar_index.json"
    
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
                $"""meta {{
  name: "%s{grammar.Metadata.Name}"
  version: "%s{grammar.Metadata.Version |> Option.defaultValue "v1.0"}"
  source: "%s{grammar.Metadata.Source |> Option.defaultValue "manual"}"
  language: "%s{grammar.Metadata.Language}"
  created: "%s{grammar.Metadata.Created |> Option.map (fun d -> d.ToString("yyyy-MM-dd HH:mm:ss")) |> Option.defaultValue ""}"
  description: "%s{grammar.Metadata.Description |> Option.defaultValue ""}"
}}

grammar {{
  LANG("%s{grammar.Metadata.Language}") {{
%s{grammar.Content.Split('\n') |> Array.map (fun line -> "    " + line) |> String.concat "\n"}
  }}
}}"""

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

            printfn $"✅ Grammar '%s{grammar.Metadata.Name}' saved to %s{filePath}"
            true
        with
        | ex ->
            printfn $"❌ Failed to save grammar '%s{grammar.Metadata.Name}': %s{ex.Message}"
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
                printfn $"❌ Failed to load grammar '%s{name}': %s{ex.Message}"
                None
        | _ -> None
    
    /// Extract inline grammar to external file
    let extractInlineGrammar name content =
        let grammar = Grammar.createInline name content
        if saveGrammar grammar then
            printfn $"✅ Extracted inline grammar '%s{name}' to external file"
            true
        else
            false
    
    /// Inline external grammar (return content for embedding)
    let inlineGrammar name =
        match loadGrammar name with
        | Some grammar ->
            Some grammar.Content
        | None ->
            printfn $"❌ Grammar '%s{name}' not found"
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
            errors.Add $"Unbalanced braces: %d{openBraces} open, %d{closeBraces} close"

        let openBrackets = grammar.Content |> Seq.filter ((=) '[') |> Seq.length
        let closeBrackets = grammar.Content |> Seq.filter ((=) ']') |> Seq.length
        if openBrackets <> closeBrackets then
            errors.Add $"Unbalanced brackets: %d{openBrackets} open, %d{closeBrackets} close"

        if errors.Count = 0 then
            Ok grammar
        else
            Error (errors |> Seq.toList)
    
    /// Generates an EBNF grammar that recognises the supplied examples
    let generateGrammarFromExamples name examples =
        if List.isEmpty examples then
            invalidArg "examples" "At least one example is required to infer a grammar."

        let sanitizedName = sanitizeIdentifier name

        let tokenised =
            examples
            |> List.map tokenizeExample

        let ruleNames =
            tokenised
            |> List.mapi (fun index _ -> $"%s{sanitizedName}_example_%02d{index + 1}")

        let startRule =
            "start = " + (ruleNames |> String.concat " | ") + " ;"

        let ruleLines =
            (tokenised, ruleNames)
            ||> List.map2 (fun tokens ruleName ->
                let body =
                    match tokens with
                    | [] -> "\"\""
                    | _ ->
                        tokens
                        |> List.map encodeToken
                        |> String.concat " , "

                $"%s{ruleName} = %s{body} ;")

        let header =
            $"(* Auto-generated grammar for '%s{name}' using %d{examples.Length} example(s) *)"

        let grammarContent =
            String.concat Environment.NewLine (header :: startRule :: ruleLines)

        Grammar.createInline name grammarContent
