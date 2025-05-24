namespace TarsEngine.FSharp.Core.Working.CodeAnalysis

open System
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Represents code analysis results.
/// </summary>
type CodeAnalysisResult = {
    Id: Id
    Timestamp: DateTime
    FilePath: string
    Language: string
    LinesOfCode: int
    Complexity: int
    QualityScore: float
    Issues: CodeIssue list
    Suggestions: string list
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a code issue.
/// </summary>
and CodeIssue = {
    Id: Id
    Severity: IssueSeverity
    Category: IssueCategory
    Message: string
    LineNumber: int option
    ColumnNumber: int option
    Rule: string option
}

/// <summary>
/// Represents issue severity levels.
/// </summary>
and IssueSeverity =
    | Info
    | Warning
    | Error
    | Critical

/// <summary>
/// Represents issue categories.
/// </summary>
and IssueCategory =
    | Style
    | Performance
    | Security
    | Maintainability
    | Reliability
    | Complexity

/// <summary>
/// Represents code metrics.
/// </summary>
type CodeMetrics = {
    LinesOfCode: int
    CyclomaticComplexity: int
    CognitiveComplexity: int
    Maintainability: float
    TestCoverage: float option
    TechnicalDebt: TimeSpan option
}

/// <summary>
/// Represents code quality assessment.
/// </summary>
type CodeQualityAssessment = {
    OverallScore: float
    Metrics: CodeMetrics
    Strengths: string list
    Weaknesses: string list
    Recommendations: string list
    ComparisonToBenchmark: float option
}

/// <summary>
/// Creates a new code issue.
/// </summary>
let createCodeIssue severity category message lineNumber =
    {
        Id = Guid.NewGuid().ToString()
        Severity = severity
        Category = category
        Message = message
        LineNumber = lineNumber
        ColumnNumber = None
        Rule = None
    }

/// <summary>
/// Creates a new code analysis result.
/// </summary>
let createAnalysisResult filePath language linesOfCode complexity qualityScore issues =
    {
        Id = Guid.NewGuid().ToString()
        Timestamp = DateTime.UtcNow
        FilePath = filePath
        Language = language
        LinesOfCode = linesOfCode
        Complexity = complexity
        QualityScore = qualityScore
        Issues = issues
        Suggestions = []
        Metadata = Map.empty
    }
