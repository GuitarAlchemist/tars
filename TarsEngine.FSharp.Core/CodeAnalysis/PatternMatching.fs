namespace TarsEngine.FSharp.Core.CodeAnalysis

/// Module for pattern matching in code
module PatternMatching =
    open System
    open System.Text.RegularExpressions
    open System.IO
    open System.Collections.Generic
    open Types
    
    /// Represents a code pattern
    type CodePattern = {
        /// The ID of the pattern
        Id: string
        /// The name of the pattern
        Name: string
        /// The description of the pattern
        Description: string
        /// The language the pattern applies to
        Language: string
        /// The pattern to match
        Pattern: string
        /// The pattern language (Regex, Literal, AST, etc.)
        PatternLanguage: string
        /// The replacement pattern (if applicable)
        Replacement: string option
        /// The explanation for the replacement
        ReplacementExplanation: string option
        /// The expected improvement from applying the pattern
        ExpectedImprovement: string option
        /// The severity of the pattern (0-1)
        Severity: float
        /// The confidence threshold for the pattern
        ConfidenceThreshold: float
        /// The impact score of the pattern
        ImpactScore: float
        /// The difficulty score of the pattern
        DifficultyScore: float
        /// The tags associated with the pattern
        Tags: string list
    }
    
    /// Represents a pattern match
    type PatternMatch = {
        /// The pattern that matched
        Pattern: CodePattern
        /// The matched text
        MatchedText: string
        /// The line number where the match was found
        LineNumber: int
        /// The column number where the match was found
        ColumnNumber: int
        /// The file path where the match was found
        FilePath: string
        /// The context around the match
        Context: string
        /// The confidence of the match
        Confidence: float
    }
    
    /// Options for pattern matching
    type PatternMatchOptions = {
        /// Whether to include line numbers
        IncludeLineNumbers: bool
        /// Whether to include column numbers
        IncludeColumnNumbers: bool
        /// Whether to include context
        IncludeContext: bool
        /// The number of context lines to include
        ContextLines: int
        /// The minimum confidence threshold
        MinConfidence: float
        /// The maximum number of matches to return
        MaxMatches: int option
    }
    
    /// Creates default pattern match options
    let defaultOptions = {
        IncludeLineNumbers = true
        IncludeColumnNumbers = true
        IncludeContext = true
        ContextLines = 2
        MinConfidence = 0.7
        MaxMatches = None
    }
    
    /// Compiles a pattern based on its pattern language
    let compilePattern (pattern: CodePattern) =
        match pattern.PatternLanguage.ToLowerInvariant() with
        | "regex" -> 
            try
                let regex = new Regex(pattern.Pattern, RegexOptions.Compiled ||| RegexOptions.Multiline)
                Ok regex
            with ex ->
                Error $"Invalid regex pattern: {ex.Message}"
        | "literal" -> Ok pattern.Pattern
        | "ast" -> Error "AST pattern matching not implemented"
        | "semantic" -> Error "Semantic pattern matching not implemented"
        | "fuzzy" -> Error "Fuzzy pattern matching not implemented"
        | "template" -> Error "Template pattern matching not implemented"
        | _ -> Error $"Unknown pattern language: {pattern.PatternLanguage}"
    
    /// Matches a regex pattern
    let matchRegexPattern (content: string) (pattern: CodePattern) (regex: Regex) (lines: string[]) (options: PatternMatchOptions) =
        let matches = regex.Matches(content)
        
        matches
        |> Seq.cast<Match>
        |> Seq.map (fun m ->
            // Calculate line and column numbers
            let lineNumber = 
                if options.IncludeLineNumbers then
                    let contentBeforeMatch = content.Substring(0, m.Index)
                    contentBeforeMatch.Split('\n').Length
                else 0
                
            let columnNumber =
                if options.IncludeColumnNumbers then
                    let contentBeforeMatch = content.Substring(0, m.Index)
                    let lastNewlineIndex = contentBeforeMatch.LastIndexOf('\n')
                    if lastNewlineIndex >= 0 then
                        m.Index - lastNewlineIndex - 1
                    else
                        m.Index
                else 0
                
            // Get context
            let context =
                if options.IncludeContext && lineNumber > 0 then
                    let startLine = max 0 (lineNumber - options.ContextLines - 1)
                    let endLine = min (lines.Length - 1) (lineNumber + options.ContextLines - 1)
                    String.Join("\n", lines[startLine..endLine])
                else ""
                
            // Calculate confidence (simplified)
            let confidence = 
                if m.Length > 10 then 0.9
                elif m.Length > 5 then 0.8
                else 0.7
                
            {
                Pattern = pattern
                MatchedText = m.Value
                LineNumber = lineNumber
                ColumnNumber = columnNumber
                FilePath = ""
                Context = context
                Confidence = confidence
            })
        |> Seq.filter (fun m -> m.Confidence >= options.MinConfidence)
        |> (fun seq -> 
            match options.MaxMatches with
            | Some max -> Seq.truncate max seq
            | None -> seq)
        |> Seq.toList
    
    /// Matches a literal pattern
    let matchLiteralPattern (content: string) (pattern: CodePattern) (literal: string) (lines: string[]) (options: PatternMatchOptions) =
        let rec findAllIndexes (str: string) (substr: string) (startIndex: int) (indexes: int list) =
            let index = str.IndexOf(substr, startIndex)
            if index = -1 then
                indexes
            else
                findAllIndexes str substr (index + substr.Length) (indexes @ [index])
                
        let indexes = findAllIndexes content literal 0 []
        
        indexes
        |> List.map (fun index ->
            // Calculate line and column numbers
            let lineNumber = 
                if options.IncludeLineNumbers then
                    let contentBeforeMatch = content.Substring(0, index)
                    contentBeforeMatch.Split('\n').Length
                else 0
                
            let columnNumber =
                if options.IncludeColumnNumbers then
                    let contentBeforeMatch = content.Substring(0, index)
                    let lastNewlineIndex = contentBeforeMatch.LastIndexOf('\n')
                    if lastNewlineIndex >= 0 then
                        index - lastNewlineIndex - 1
                    else
                        index
                else 0
                
            // Get context
            let context =
                if options.IncludeContext && lineNumber > 0 then
                    let startLine = max 0 (lineNumber - options.ContextLines - 1)
                    let endLine = min (lines.Length - 1) (lineNumber + options.ContextLines - 1)
                    String.Join("\n", lines[startLine..endLine])
                else ""
                
            // Calculate confidence (exact match = high confidence)
            let confidence = 0.95
                
            {
                Pattern = pattern
                MatchedText = literal
                LineNumber = lineNumber
                ColumnNumber = columnNumber
                FilePath = ""
                Context = context
                Confidence = confidence
            })
        |> List.filter (fun m -> m.Confidence >= options.MinConfidence)
        |> (fun list -> 
            match options.MaxMatches with
            | Some max -> List.truncate max list
            | None -> list)
    
    /// Finds patterns in content
    let findPatterns (content: string) (patterns: CodePattern list) (language: string) (options: PatternMatchOptions) =
        let lines = content.Split('\n')
        
        patterns
        |> List.filter (fun p -> p.Language.Equals(language, StringComparison.OrdinalIgnoreCase) || p.Language.Equals("any", StringComparison.OrdinalIgnoreCase))
        |> List.collect (fun pattern ->
            match compilePattern pattern with
            | Ok compiledPattern ->
                match pattern.PatternLanguage.ToLowerInvariant() with
                | "regex" -> matchRegexPattern content pattern (compiledPattern :?> Regex) lines options
                | "literal" -> matchLiteralPattern content pattern (compiledPattern :?> string) lines options
                | _ -> [] // Other pattern types not implemented yet
            | Error _ -> [])
    
    /// Finds patterns in a file
    let findPatternsInFile (filePath: string) (patterns: CodePattern list) (options: PatternMatchOptions) =
        try
            let content = File.ReadAllText(filePath)
            let language = 
                match Path.GetExtension(filePath).ToLowerInvariant() with
                | ".cs" -> "csharp"
                | ".fs" | ".fsx" -> "fsharp"
                | ".js" -> "javascript"
                | ".ts" -> "typescript"
                | ".py" -> "python"
                | ".java" -> "java"
                | _ -> "unknown"
                
            let matches = findPatterns content patterns language options
            
            // Add file path to matches
            matches
            |> List.map (fun m -> { m with FilePath = filePath })
        with
        | ex -> 
            printfn "Error analyzing file %s: %s" filePath ex.Message
            []
    
    /// Finds patterns in a directory
    let findPatternsInDirectory (directoryPath: string) (patterns: CodePattern list) (options: PatternMatchOptions) (fileExtensions: string list) (excludeDirs: string list) =
        try
            let isExcluded (path: string) =
                excludeDirs
                |> List.exists (fun exclude -> path.Contains(exclude))
                
            let isIncluded (path: string) =
                fileExtensions
                |> List.exists (fun ext -> Path.GetExtension(path).Equals(ext, StringComparison.OrdinalIgnoreCase))
                
            Directory.GetFiles(directoryPath, "*.*", SearchOption.AllDirectories)
            |> Array.filter (fun path -> not (isExcluded path) && isIncluded path)
            |> Array.collect (fun path -> findPatternsInFile path patterns options |> List.toArray)
            |> Array.toList
        with
        | ex -> 
            printfn "Error analyzing directory %s: %s" directoryPath ex.Message
            []
    
    /// Calculates the similarity between two strings
    let calculateSimilarity (source: string) (target: string) =
        // Simplified Levenshtein distance implementation
        let m = source.Length
        let n = target.Length
        
        // Handle edge cases
        if m = 0 then 
            if n = 0 then 1.0 else 0.0
        elif n = 0 then 
            0.0
        else
            // Initialize distance matrix
            let distance = Array2D.create (m + 1) (n + 1) 0
            
            // Initialize first row and column
            for i in 0..m do
                distance[i, 0] <- i
            for j in 0..n do
                distance[0, j] <- j
                
            // Fill the matrix
            for i in 1..m do
                for j in 1..n do
                    let cost = if source[i-1] = target[j-1] then 0 else 1
                    distance[i, j] <- min (min (distance[i-1, j] + 1) (distance[i, j-1] + 1)) (distance[i-1, j-1] + cost)
                    
            // Calculate similarity
            let maxLength = max m n
            1.0 - (float distance[m, n] / float maxLength)
    
    /// Finds similar patterns
    let findSimilarPatterns (content: string) (patterns: CodePattern list) (language: string) (minSimilarity: float) (maxResults: int) =
        patterns
        |> List.filter (fun p -> p.Language.Equals(language, StringComparison.OrdinalIgnoreCase) || p.Language.Equals("any", StringComparison.OrdinalIgnoreCase))
        |> List.map (fun pattern ->
            let similarity = calculateSimilarity content pattern.Pattern
            (pattern, similarity))
        |> List.filter (fun (_, similarity) -> similarity >= minSimilarity)
        |> List.sortByDescending snd
        |> List.truncate maxResults
