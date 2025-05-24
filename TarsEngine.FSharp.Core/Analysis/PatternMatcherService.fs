namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of IPatternMatcherService.
/// </summary>
type PatternMatcherService(logger: ILogger<PatternMatcherService>, patterns: CodePattern list) =
    
    /// <summary>
    /// Gets the language for a file based on its extension.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>The language of the file.</returns>
    member private _.GetLanguageForFile(filePath: string) =
        match Path.GetExtension(filePath).ToLowerInvariant() with
        | ".cs" -> "csharp"
        | ".fs" | ".fsx" | ".fsi" -> "fsharp"
        | ".js" -> "javascript"
        | ".ts" -> "typescript"
        | ".py" -> "python"
        | ".java" -> "java"
        | ".rb" -> "ruby"
        | ".go" -> "go"
        | ".php" -> "php"
        | ".swift" -> "swift"
        | ".kt" | ".kts" -> "kotlin"
        | ".scala" -> "scala"
        | ".rs" -> "rust"
        | ".c" | ".h" -> "c"
        | ".cpp" | ".hpp" | ".cc" | ".hh" -> "cpp"
        | ".m" | ".mm" -> "objectivec"
        | ".pl" | ".pm" -> "perl"
        | ".r" -> "r"
        | ".sh" | ".bash" -> "shell"
        | ".sql" -> "sql"
        | ".xml" -> "xml"
        | ".json" -> "json"
        | ".yaml" | ".yml" -> "yaml"
        | ".html" | ".htm" -> "html"
        | ".css" -> "css"
        | ".md" -> "markdown"
        | _ -> "unknown"
    
    /// <summary>
    /// Gets patterns for a specific language.
    /// </summary>
    /// <param name="language">The language to get patterns for.</param>
    /// <returns>The list of patterns for the language.</returns>
    member private _.GetPatternsForLanguage(language: string) =
        patterns |> List.filter (fun p -> p.Language.Equals(language, StringComparison.OrdinalIgnoreCase))
    
    /// <summary>
    /// Finds patterns in the provided content.
    /// </summary>
    /// <param name="content">The code content to analyze.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="options">Optional matching options.</param>
    /// <returns>The list of pattern matches.</returns>
    member this.FindPatternsAsync(content: string, language: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Finding patterns in {Language} code", language)
                
                // Get patterns for the language
                let languagePatterns = this.GetPatternsForLanguage(language)
                
                // Find matches for each pattern
                let matches = ResizeArray<PatternMatch>()
                
                for pattern in languagePatterns do
                    try
                        // Create a regex from the pattern template
                        let regex = new Regex(pattern.Template, RegexOptions.Multiline ||| RegexOptions.Singleline)
                        
                        // Find matches
                        let regexMatches = regex.Matches(content)
                        
                        for m in regexMatches do
                            let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                            let endLineNumber = lineNumber + m.Value.Split('\n').Length - 1
                            
                            // Calculate confidence based on match length and pattern specificity
                            let confidence = 
                                let matchLength = m.Value.Length
                                let contentLength = content.Length
                                let patternLength = pattern.Template.Length
                                
                                // Longer matches and more specific patterns have higher confidence
                                let lengthFactor = Math.Min(1.0, (float matchLength) / 100.0)
                                let specificityFactor = Math.Min(1.0, (float patternLength) / 100.0)
                                
                                (lengthFactor + specificityFactor) / 2.0
                            
                            // Create a pattern match
                            let patternMatch = {
                                PatternName = pattern.Name
                                Description = pattern.Description
                                StartLine = lineNumber
                                EndLine = endLineNumber
                                CodeSnippet = m.Value
                                Confidence = confidence
                                AdditionalInfo = Map.empty
                            }
                            
                            matches.Add(patternMatch)
                    with
                    | ex ->
                        logger.LogError(ex, "Error finding matches for pattern {PatternName}", pattern.Name)
                
                return matches |> Seq.toList
            with
            | ex ->
                logger.LogError(ex, "Error finding patterns in {Language} code", language)
                return []
        }
    
    /// <summary>
    /// Finds patterns in a file.
    /// </summary>
    /// <param name="filePath">The path to the file to analyze.</param>
    /// <param name="options">Optional matching options.</param>
    /// <returns>The list of pattern matches.</returns>
    member this.FindPatternsInFileAsync(filePath: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Finding patterns in file: {FilePath}", filePath)
                
                // Get the language for the file
                let language = this.GetLanguageForFile(filePath)
                
                // Read the file content
                let content = File.ReadAllText(filePath)
                
                // Find patterns in the content
                let! matches = this.FindPatternsAsync(content, language, ?options = options)
                
                return matches
            with
            | ex ->
                logger.LogError(ex, "Error finding patterns in file: {FilePath}", filePath)
                return []
        }
    
    /// <summary>
    /// Finds patterns in a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory to analyze.</param>
    /// <param name="recursive">Whether to analyze subdirectories.</param>
    /// <param name="filePattern">The pattern to match files to analyze.</param>
    /// <param name="options">Optional matching options.</param>
    /// <returns>The list of pattern matches grouped by file.</returns>
    member this.FindPatternsInDirectoryAsync(directoryPath: string, ?recursive: bool, ?filePattern: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Finding patterns in directory: {DirectoryPath}", directoryPath)
                
                // Get the search option
                let searchOption = 
                    if recursive.GetValueOrDefault(false) then
                        SearchOption.AllDirectories
                    else
                        SearchOption.TopDirectoryOnly
                
                // Get the file pattern
                let pattern = filePattern.GetValueOrDefault("*.*")
                
                // Get all files matching the pattern
                let files = Directory.GetFiles(directoryPath, pattern, searchOption)
                
                // Find patterns in each file
                let results = Dictionary<string, PatternMatch list>()
                
                for file in files do
                    let! matches = this.FindPatternsInFileAsync(file, ?options = options)
                    if not (List.isEmpty matches) then
                        results.Add(file, matches)
                
                return results |> Seq.map (fun kvp -> (kvp.Key, kvp.Value)) |> Map.ofSeq
            with
            | ex ->
                logger.LogError(ex, "Error finding patterns in directory: {DirectoryPath}", directoryPath)
                return Map.empty
        }
    
    /// <summary>
    /// Calculates the similarity between two code snippets.
    /// </summary>
    /// <param name="source">The source code snippet.</param>
    /// <param name="target">The target code snippet.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <returns>The similarity score (0.0 to 1.0).</returns>
    member _.CalculateSimilarityAsync(source: string, target: string, language: string) =
        task {
            try
                logger.LogInformation("Calculating similarity between code snippets")
                
                // Normalize the code snippets
                let normalizeCode (code: string) =
                    // Remove comments
                    let withoutComments = 
                        match language with
                        | "csharp" | "java" | "javascript" | "typescript" | "c" | "cpp" ->
                            // Remove C-style comments
                            let withoutBlockComments = Regex.Replace(code, @"/\*[\s\S]*?\*/", "")
                            Regex.Replace(withoutBlockComments, @"//.*$", "", RegexOptions.Multiline)
                        | "fsharp" ->
                            // Remove F# comments
                            let withoutBlockComments = Regex.Replace(code, @"\(\*[\s\S]*?\*\)", "")
                            Regex.Replace(withoutBlockComments, @"//.*$", "", RegexOptions.Multiline)
                        | "python" ->
                            // Remove Python comments
                            let withoutBlockComments = Regex.Replace(code, @"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"", "")
                            Regex.Replace(withoutBlockComments, @"#.*$", "", RegexOptions.Multiline)
                        | _ ->
                            // Default to C-style comments
                            let withoutBlockComments = Regex.Replace(code, @"/\*[\s\S]*?\*/", "")
                            Regex.Replace(withoutBlockComments, @"//.*$", "", RegexOptions.Multiline)
                    
                    // Remove whitespace
                    let withoutWhitespace = Regex.Replace(withoutComments, @"\s+", " ")
                    
                    // Remove string literals
                    let withoutStrings = Regex.Replace(withoutWhitespace, @"""[^""]*""", "\"\"")
                    
                    // Remove numeric literals
                    let withoutNumbers = Regex.Replace(withoutStrings, @"\b\d+\b", "0")
                    
                    withoutNumbers.Trim()
                
                let normalizedSource = normalizeCode source
                let normalizedTarget = normalizeCode target
                
                // Calculate Levenshtein distance
                let levenshteinDistance (s1: string) (s2: string) =
                    let len1 = s1.Length
                    let len2 = s2.Length
                    
                    // Initialize the distance matrix
                    let matrix = Array2D.zeroCreate<int> (len1 + 1) (len2 + 1)
                    
                    // Initialize the first row and column
                    for i in 0..len1 do
                        matrix.[i, 0] <- i
                    
                    for j in 0..len2 do
                        matrix.[0, j] <- j
                    
                    // Fill the rest of the matrix
                    for i in 1..len1 do
                        for j in 1..len2 do
                            let cost = if s1.[i - 1] = s2.[j - 1] then 0 else 1
                            
                            matrix.[i, j] <- 
                                [
                                    matrix.[i - 1, j] + 1      // Deletion
                                    matrix.[i, j - 1] + 1      // Insertion
                                    matrix.[i - 1, j - 1] + cost  // Substitution
                                ]
                                |> List.min
                    
                    // Return the distance
                    matrix.[len1, len2]
                
                // Calculate the similarity score
                let distance = levenshteinDistance normalizedSource normalizedTarget
                let maxLength = Math.Max(normalizedSource.Length, normalizedTarget.Length)
                
                let similarity = 
                    if maxLength = 0 then
                        1.0 // Both strings are empty, so they're identical
                    else
                        1.0 - (float distance) / (float maxLength)
                
                return similarity
            with
            | ex ->
                logger.LogError(ex, "Error calculating similarity between code snippets")
                return 0.0
        }
    
    /// <summary>
    /// Finds similar patterns to the provided code.
    /// </summary>
    /// <param name="content">The code content to find similar patterns for.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="minSimilarity">The minimum similarity score (0.0 to 1.0).</param>
    /// <param name="maxResults">The maximum number of results to return.</param>
    /// <returns>The list of similar patterns with their similarity scores.</returns>
    member this.FindSimilarPatternsAsync(content: string, language: string, ?minSimilarity: float, ?maxResults: int) =
        task {
            try
                logger.LogInformation("Finding similar patterns to {Language} code", language)
                
                // Get patterns for the language
                let languagePatterns = this.GetPatternsForLanguage(language)
                
                // Calculate similarity for each pattern
                let similarities = ResizeArray<CodePattern * float>()
                
                for pattern in languagePatterns do
                    let! similarity = this.CalculateSimilarityAsync(content, pattern.Template, language)
                    
                    // Only include patterns with similarity above the threshold
                    let threshold = minSimilarity.GetValueOrDefault(0.7)
                    if similarity >= threshold then
                        similarities.Add((pattern, similarity))
                
                // Sort by similarity (descending) and take the top results
                let limit = maxResults.GetValueOrDefault(10)
                let topResults = 
                    similarities
                    |> Seq.sortByDescending snd
                    |> Seq.take (Math.Min(limit, similarities.Count))
                    |> Seq.toList
                
                return topResults
            with
            | ex ->
                logger.LogError(ex, "Error finding similar patterns to {Language} code", language)
                return []
        }
    
    interface IPatternMatcherService with
        member this.FindPatternsAsync(content, language, ?options) = this.FindPatternsAsync(content, language, ?options = options)
        member this.FindPatternsInFileAsync(filePath, ?options) = this.FindPatternsInFileAsync(filePath, ?options = options)
        member this.FindPatternsInDirectoryAsync(directoryPath, ?recursive, ?filePattern, ?options) = this.FindPatternsInDirectoryAsync(directoryPath, ?recursive = recursive, ?filePattern = filePattern, ?options = options)
        member this.CalculateSimilarityAsync(source, target, language) = this.CalculateSimilarityAsync(source, target, language)
        member this.FindSimilarPatternsAsync(content, language, ?minSimilarity, ?maxResults) = this.FindSimilarPatternsAsync(content, language, ?minSimilarity = minSimilarity, ?maxResults = maxResults)
