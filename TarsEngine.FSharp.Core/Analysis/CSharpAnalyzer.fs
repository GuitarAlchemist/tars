namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of ILanguageAnalyzer for C# language.
/// </summary>
type CSharpAnalyzer(logger: ILogger<CSharpAnalyzer>, 
                    structureExtractor: CSharpStructureExtractor,
                    securityIssueDetector: ISecurityIssueDetector,
                    performanceIssueDetector: IPerformanceIssueDetector,
                    complexityIssueDetector: IComplexityIssueDetector,
                    styleIssueDetector: IStyleIssueDetector) =
    
    /// <summary>
    /// Gets the language supported by this analyzer.
    /// </summary>
    member _.Language = "csharp"
    
    /// <summary>
    /// Analyzes code content.
    /// </summary>
    /// <param name="content">The code content to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    member _.AnalyzeAsync(content: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Analyzing C# code")
                
                // Extract structures
                let structures = structureExtractor.ExtractStructures(content)
                
                // Detect issues
                let securityIssues = securityIssueDetector.DetectIssues(content)
                let performanceIssues = performanceIssueDetector.DetectIssues(content)
                let complexityIssues = complexityIssueDetector.DetectIssues(content)
                let styleIssues = styleIssueDetector.DetectIssues(content)
                
                // Combine all issues
                let allIssues = 
                    List.concat [
                        securityIssues
                        performanceIssues
                        complexityIssues
                        styleIssues
                    ]
                
                // Calculate metrics
                let metrics = Map.empty
                    .Add("LineCount", content.Split('\n').Length :> obj)
                    .Add("CharacterCount", content.Length :> obj)
                    .Add("ClassCount", structures |> List.filter (fun s -> s.StructureType = "class") |> List.length :> obj)
                    .Add("MethodCount", structures |> List.filter (fun s -> s.StructureType = "method") |> List.length :> obj)
                    .Add("PropertyCount", structures |> List.filter (fun s -> s.StructureType = "property") |> List.length :> obj)
                    .Add("FieldCount", structures |> List.filter (fun s -> s.StructureType = "field") |> List.length :> obj)
                    .Add("InterfaceCount", structures |> List.filter (fun s -> s.StructureType = "interface") |> List.length :> obj)
                    .Add("EnumCount", structures |> List.filter (fun s -> s.StructureType = "enum") |> List.length :> obj)
                    .Add("StructCount", structures |> List.filter (fun s -> s.StructureType = "struct") |> List.length :> obj)
                    .Add("DelegateCount", structures |> List.filter (fun s -> s.StructureType = "delegate") |> List.length :> obj)
                    .Add("EventCount", structures |> List.filter (fun s -> s.StructureType = "event") |> List.length :> obj)
                    .Add("NamespaceCount", structures |> List.filter (fun s -> s.StructureType = "namespace") |> List.length :> obj)
                    .Add("UsingCount", structures |> List.filter (fun s -> s.StructureType = "using") |> List.length :> obj)
                    .Add("AttributeCount", structures |> List.filter (fun s -> s.StructureType = "attribute") |> List.length :> obj)
                    .Add("CommentCount", Regex.Matches(content, @"//.*$|/\*[\s\S]*?\*/", RegexOptions.Multiline).Count :> obj)
                    .Add("TodoCount", Regex.Matches(content, @"//\s*TODO:", RegexOptions.IgnoreCase ||| RegexOptions.Multiline).Count :> obj)
                    .Add("FixmeCount", Regex.Matches(content, @"//\s*FIXME:", RegexOptions.IgnoreCase ||| RegexOptions.Multiline).Count :> obj)
                    .Add("HackCount", Regex.Matches(content, @"//\s*HACK:", RegexOptions.IgnoreCase ||| RegexOptions.Multiline).Count :> obj)
                    .Add("NoteCount", Regex.Matches(content, @"//\s*NOTE:", RegexOptions.IgnoreCase ||| RegexOptions.Multiline).Count :> obj)
                    .Add("WarningCount", Regex.Matches(content, @"//\s*WARNING:", RegexOptions.IgnoreCase ||| RegexOptions.Multiline).Count :> obj)
                
                // Create the analysis result
                let result = {
                    FilePath = None
                    Language = "csharp"
                    Issues = allIssues
                    Structures = structures
                    Patterns = []
                    Metrics = metrics
                    AdditionalInfo = Map.empty
                }
                
                return result
            with
            | ex ->
                logger.LogError(ex, "Error analyzing C# code")
                return {
                    FilePath = None
                    Language = "csharp"
                    Issues = [{
                        IssueType = CodeIssueType.Other
                        Severity = IssueSeverity.Error
                        Message = $"Error analyzing code: {ex.Message}"
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
    /// Analyzes a file.
    /// </summary>
    /// <param name="filePath">The path to the file to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    member this.AnalyzeFileAsync(filePath: string, ?options: Map<string, string>) =
        task {
            try
                logger.LogInformation("Analyzing C# file: {FilePath}", filePath)
                
                // Read the file content
                let content = File.ReadAllText(filePath)
                
                // Analyze the content
                let! result = this.AnalyzeAsync(content, ?options = options)
                
                // Update the result with the file path
                return { result with FilePath = Some filePath }
            with
            | ex ->
                logger.LogError(ex, "Error analyzing C# file: {FilePath}", filePath)
                return {
                    FilePath = Some filePath
                    Language = "csharp"
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
    /// Gets the supported analysis options.
    /// </summary>
    /// <returns>A dictionary of option names to descriptions.</returns>
    member _.GetSupportedOptions() =
        let options = Dictionary<string, string>()
        options.Add("IncludeSecurity", "Include security issues in the analysis")
        options.Add("IncludePerformance", "Include performance issues in the analysis")
        options.Add("IncludeComplexity", "Include complexity issues in the analysis")
        options.Add("IncludeStyle", "Include style issues in the analysis")
        options.Add("MaxIssues", "Maximum number of issues to return")
        options.Add("MinSeverity", "Minimum severity of issues to return")
        options :> IDictionary<string, string>
    
    interface ILanguageAnalyzer with
        member this.Language = this.Language
        member this.AnalyzeAsync(content, ?options) = this.AnalyzeAsync(content, ?options = options)
        member this.AnalyzeFileAsync(filePath, ?options) = this.AnalyzeFileAsync(filePath, ?options = options)
        member this.GetSupportedOptions() = this.GetSupportedOptions()
