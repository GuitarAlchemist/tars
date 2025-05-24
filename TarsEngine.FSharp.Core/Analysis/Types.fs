namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Collections.Generic
open System.Threading.Tasks

/// <summary>
/// Represents the severity of a code issue.
/// </summary>
type IssueSeverity =
    | Info = 0
    | Warning = 1
    | Error = 2
    | Critical = 3

/// <summary>
/// Represents the type of a code issue.
/// </summary>
type CodeIssueType =
    | Style = 0
    | Performance = 1
    | Security = 2
    | Complexity = 3
    | Maintainability = 4
    | BestPractice = 5
    | Compatibility = 6
    | Reliability = 7
    | Documentation = 8
    | Other = 9

/// <summary>
/// Represents a code issue detected during analysis.
/// </summary>
type CodeIssue = {
    /// <summary>
    /// The type of the issue.
    /// </summary>
    IssueType: CodeIssueType
    
    /// <summary>
    /// The severity of the issue.
    /// </summary>
    Severity: IssueSeverity
    
    /// <summary>
    /// The message describing the issue.
    /// </summary>
    Message: string
    
    /// <summary>
    /// The line number where the issue was detected.
    /// </summary>
    LineNumber: int option
    
    /// <summary>
    /// The column number where the issue was detected.
    /// </summary>
    ColumnNumber: int option
    
    /// <summary>
    /// The file path where the issue was detected.
    /// </summary>
    FilePath: string option
    
    /// <summary>
    /// The code snippet where the issue was detected.
    /// </summary>
    CodeSnippet: string option
    
    /// <summary>
    /// The suggested fix for the issue.
    /// </summary>
    SuggestedFix: string option
    
    /// <summary>
    /// Additional information about the issue.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a code structure extracted from source code.
/// </summary>
type CodeStructure = {
    /// <summary>
    /// The name of the structure.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The type of the structure (e.g., class, method, property).
    /// </summary>
    StructureType: string
    
    /// <summary>
    /// The parent structure, if any.
    /// </summary>
    Parent: CodeStructure option
    
    /// <summary>
    /// The child structures.
    /// </summary>
    Children: CodeStructure list
    
    /// <summary>
    /// The start line of the structure.
    /// </summary>
    StartLine: int
    
    /// <summary>
    /// The end line of the structure.
    /// </summary>
    EndLine: int
    
    /// <summary>
    /// The modifiers of the structure (e.g., public, private, static).
    /// </summary>
    Modifiers: string list
    
    /// <summary>
    /// The return type of the structure, if applicable.
    /// </summary>
    ReturnType: string option
    
    /// <summary>
    /// The parameters of the structure, if applicable.
    /// </summary>
    Parameters: (string * string) list
    
    /// <summary>
    /// Additional properties of the structure.
    /// </summary>
    Properties: Map<string, string>
}

/// <summary>
/// Represents a pattern match found in code.
/// </summary>
type PatternMatch = {
    /// <summary>
    /// The name of the pattern.
    /// </summary>
    PatternName: string
    
    /// <summary>
    /// The description of the pattern.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The start line of the match.
    /// </summary>
    StartLine: int
    
    /// <summary>
    /// The end line of the match.
    /// </summary>
    EndLine: int
    
    /// <summary>
    /// The matched code snippet.
    /// </summary>
    CodeSnippet: string
    
    /// <summary>
    /// The confidence level of the match (0.0 to 1.0).
    /// </summary>
    Confidence: float
    
    /// <summary>
    /// Additional information about the match.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a code pattern.
/// </summary>
type CodePattern = {
    /// <summary>
    /// The name of the pattern.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the pattern.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The language of the pattern.
    /// </summary>
    Language: string
    
    /// <summary>
    /// The pattern template.
    /// </summary>
    Template: string
    
    /// <summary>
    /// The pattern category.
    /// </summary>
    Category: string
    
    /// <summary>
    /// The tags associated with the pattern.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// Additional information about the pattern.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents the result of a code analysis.
/// </summary>
type CodeAnalysisResult = {
    /// <summary>
    /// The file path that was analyzed.
    /// </summary>
    FilePath: string option
    
    /// <summary>
    /// The language of the analyzed code.
    /// </summary>
    Language: string
    
    /// <summary>
    /// The issues found during analysis.
    /// </summary>
    Issues: CodeIssue list
    
    /// <summary>
    /// The structures extracted from the code.
    /// </summary>
    Structures: CodeStructure list
    
    /// <summary>
    /// The patterns found in the code.
    /// </summary>
    Patterns: PatternMatch list
    
    /// <summary>
    /// The metrics calculated for the code.
    /// </summary>
    Metrics: Map<string, obj>
    
    /// <summary>
    /// Additional information about the analysis.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a readability metric.
/// </summary>
type ReadabilityMetric = {
    /// <summary>
    /// The name of the metric.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The value of the metric.
    /// </summary>
    Value: float
    
    /// <summary>
    /// The description of the metric.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The scope of the metric (e.g., file, method, class).
    /// </summary>
    Scope: string
    
    /// <summary>
    /// The target of the metric (e.g., file path, method name).
    /// </summary>
    Target: string
    
    /// <summary>
    /// Additional information about the metric.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a complexity metric.
/// </summary>
type ComplexityMetric = {
    /// <summary>
    /// The name of the metric.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The value of the metric.
    /// </summary>
    Value: float
    
    /// <summary>
    /// The description of the metric.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The scope of the metric (e.g., file, method, class).
    /// </summary>
    Scope: string
    
    /// <summary>
    /// The target of the metric (e.g., file path, method name).
    /// </summary>
    Target: string
    
    /// <summary>
    /// Additional information about the metric.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a maintainability metric.
/// </summary>
type MaintainabilityMetric = {
    /// <summary>
    /// The name of the metric.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The value of the metric.
    /// </summary>
    Value: float
    
    /// <summary>
    /// The description of the metric.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The scope of the metric (e.g., file, method, class).
    /// </summary>
    Scope: string
    
    /// <summary>
    /// The target of the metric (e.g., file path, method name).
    /// </summary>
    Target: string
    
    /// <summary>
    /// Additional information about the metric.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a Halstead complexity metric.
/// </summary>
type HalsteadMetric = {
    /// <summary>
    /// The name of the metric.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The value of the metric.
    /// </summary>
    Value: float
    
    /// <summary>
    /// The description of the metric.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The scope of the metric (e.g., file, method, class).
    /// </summary>
    Scope: string
    
    /// <summary>
    /// The target of the metric (e.g., file path, method name).
    /// </summary>
    Target: string
    
    /// <summary>
    /// Additional information about the metric.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents the type of readability to analyze.
/// </summary>
type ReadabilityType =
    | FleschKincaid = 0
    | GunningFog = 1
    | ColemanLiau = 2
    | AutomatedReadability = 3
    | SMOG = 4
    | Custom = 5
