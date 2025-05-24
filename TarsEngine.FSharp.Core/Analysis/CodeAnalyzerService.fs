namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Collections.Generic
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of ICodeAnalyzerService.
/// </summary>
type CodeAnalyzerService(logger: ILogger<CodeAnalyzerService>, analyzers: ILanguageAnalyzer seq) =
    
    /// <summary>
    /// Gets the language analyzer for the specified language.
    /// </summary>
    /// <param name="language">The language to get the analyzer for.</param>
    /// <returns>The language analyzer, if found.</returns>
    member private _.GetAnalyzerForLanguage(language: string) =
        analyzers 
        |> Seq.tryFind (fun a -> a.Language.Equals(language, StringComparison.OrdinalIgnoreCase))
    
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
    /// Analyzes a file.
    /// </summary>
    /// <param name="filePath">The path to the file to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    member this.AnalyzeFileAsync(filePath: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Analyzing file: {FilePath}", filePath)
                
                // Get the language for the file
                let language = this.GetLanguageForFile(filePath)
                
                // Get the analyzer for the language
                match this.GetAnalyzerForLanguage(language) with
                | Some analyzer ->
                    // Analyze the file
                    return! analyzer.AnalyzeFileAsync(filePath, ?options = options)
                | None ->
                    logger.LogWarning("No analyzer found for language: {Language}", language)
                    return {
                        FilePath = Some filePath
                        Language = language
                        Issues = [{
                            IssueType = CodeIssueType.Other
                            Severity = IssueSeverity.Error
                            Message = $"No analyzer found for language: {language}"
                            LineNumber = None
                            ColumnNumber = None
                            FilePath = Some filePath
                            CodeSnippet = None
                            SuggestedFix = None
                            AdditionalInfo = Map.empty
                        }]
                        Structures = []
                        Patterns = []
                        Metrics = Map.empty
                        AdditionalInfo = Map.empty
                    }
            with
            | ex ->
                logger.LogError(ex, "Error analyzing file: {FilePath}", filePath)
                return {
                    FilePath = Some filePath
                    Language = "unknown"
                    Issues = [{
                        IssueType = CodeIssueType.Other
                        Severity = IssueSeverity.Error
                        Message = $"Error analyzing file: {ex.Message}"
                        LineNumber = None
                        ColumnNumber = None
                        FilePath = Some filePath
                        CodeSnippet = None
                        SuggestedFix = None
                        AdditionalInfo = Map.empty
                    }]
                    Structures = []
                    Patterns = []
                    Metrics = Map.empty
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Analyzes a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory to analyze.</param>
    /// <param name="recursive">Whether to analyze subdirectories.</param>
    /// <param name="filePattern">The pattern to match files to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis results.</returns>
    member this.AnalyzeDirectoryAsync(directoryPath: string, ?recursive: bool, ?filePattern: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Analyzing directory: {DirectoryPath}", directoryPath)
                
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
                
                // Analyze each file
                let results = ResizeArray<CodeAnalysisResult>()
                for file in files do
                    let! result = this.AnalyzeFileAsync(file, ?options = options)
                    results.Add(result)
                
                return results |> Seq.toList
            with
            | ex ->
                logger.LogError(ex, "Error analyzing directory: {DirectoryPath}", directoryPath)
                return [{
                    FilePath = Some directoryPath
                    Language = "unknown"
                    Issues = [{
                        IssueType = CodeIssueType.Other
                        Severity = IssueSeverity.Error
                        Message = $"Error analyzing directory: {ex.Message}"
                        LineNumber = None
                        ColumnNumber = None
                        FilePath = Some directoryPath
                        CodeSnippet = None
                        SuggestedFix = None
                        AdditionalInfo = Map.empty
                    }]
                    Structures = []
                    Patterns = []
                    Metrics = Map.empty
                    AdditionalInfo = Map.empty
                }]
        }
    
    /// <summary>
    /// Analyzes code content.
    /// </summary>
    /// <param name="content">The code content to analyze.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    member this.AnalyzeContentAsync(content: string, language: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Analyzing code content")
                
                // Get the analyzer for the language
                match this.GetAnalyzerForLanguage(language) with
                | Some analyzer ->
                    // Analyze the content
                    return! analyzer.AnalyzeAsync(content, ?options = options)
                | None ->
                    logger.LogWarning("No analyzer found for language: {Language}", language)
                    return {
                        FilePath = None
                        Language = language
                        Issues = [{
                            IssueType = CodeIssueType.Other
                            Severity = IssueSeverity.Error
                            Message = $"No analyzer found for language: {language}"
                            LineNumber = None
                            ColumnNumber = None
                            FilePath = None
                            CodeSnippet = None
                            SuggestedFix = None
                            AdditionalInfo = Map.empty
                        }]
                        Structures = []
                        Patterns = []
                        Metrics = Map.empty
                        AdditionalInfo = Map.empty
                    }
            with
            | ex ->
                logger.LogError(ex, "Error analyzing code content")
                return {
                    FilePath = None
                    Language = language
                    Issues = [{
                        IssueType = CodeIssueType.Other
                        Severity = IssueSeverity.Error
                        Message = $"Error analyzing code content: {ex.Message}"
                        LineNumber = None
                        ColumnNumber = None
                        FilePath = None
                        CodeSnippet = None
                        SuggestedFix = None
                        AdditionalInfo = Map.empty
                    }]
                    Structures = []
                    Patterns = []
                    Metrics = Map.empty
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Gets the supported languages for analysis.
    /// </summary>
    /// <returns>The list of supported languages.</returns>
    member _.GetSupportedLanguagesAsync() =
        task {
            return analyzers |> Seq.map (fun a -> a.Language) |> Seq.toList
        }
    
    /// <summary>
    /// Gets the issues for a specific file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <param name="issueTypes">The types of issues to get.</param>
    /// <param name="minSeverity">The minimum severity of issues to get.</param>
    /// <param name="options">Optional filtering options.</param>
    /// <returns>The list of issues.</returns>
    member this.GetIssuesForFileAsync(filePath: string, ?issueTypes: CodeIssueType list, ?minSeverity: IssueSeverity, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Getting issues for file: {FilePath}", filePath)
                
                // Analyze the file
                let! result = this.AnalyzeFileAsync(filePath, ?options = options)
                
                // Filter issues by type
                let filteredByType = 
                    match issueTypes with
                    | Some types -> result.Issues |> List.filter (fun i -> types |> List.contains i.IssueType)
                    | None -> result.Issues
                
                // Filter issues by severity
                let filteredBySeverity = 
                    match minSeverity with
                    | Some severity -> filteredByType |> List.filter (fun i -> i.Severity >= severity)
                    | None -> filteredByType
                
                return filteredBySeverity
            with
            | ex ->
                logger.LogError(ex, "Error getting issues for file: {FilePath}", filePath)
                return [{
                    IssueType = CodeIssueType.Other
                    Severity = IssueSeverity.Error
                    Message = $"Error getting issues for file: {ex.Message}"
                    LineNumber = None
                    ColumnNumber = None
                    FilePath = Some filePath
                    CodeSnippet = None
                    SuggestedFix = None
                    AdditionalInfo = Map.empty
                }]
        }
    
    interface ICodeAnalyzerService with
        member this.AnalyzeFileAsync(filePath, ?options) = this.AnalyzeFileAsync(filePath, ?options = options)
        member this.AnalyzeDirectoryAsync(directoryPath, ?recursive, ?filePattern, ?options) = this.AnalyzeDirectoryAsync(directoryPath, ?recursive = recursive, ?filePattern = filePattern, ?options = options)
        member this.AnalyzeContentAsync(content, language, ?options) = this.AnalyzeContentAsync(content, language, ?options = options)
        member this.GetSupportedLanguagesAsync() = this.GetSupportedLanguagesAsync()
        member this.GetIssuesForFileAsync(filePath, ?issueTypes, ?minSeverity, ?options) = this.GetIssuesForFileAsync(filePath, ?issueTypes = issueTypes, ?minSeverity = minSeverity, ?options = options)
