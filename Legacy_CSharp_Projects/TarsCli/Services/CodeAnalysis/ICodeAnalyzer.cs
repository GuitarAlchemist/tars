namespace TarsCli.Services.CodeAnalysis;

/// <summary>
/// Interface for code analyzers
/// </summary>
public interface ICodeAnalyzer
{
    /// <summary>
    /// Analyzes a file for improvement opportunities
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="fileContent">Content of the file to analyze</param>
    /// <returns>Analysis result</returns>
    Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath, string fileContent);

    /// <summary>
    /// Gets the supported file extensions for this analyzer
    /// </summary>
    /// <returns>List of supported file extensions</returns>
    IEnumerable<string> GetSupportedFileExtensions();
}

/// <summary>
/// Result of code analysis
/// </summary>
public class CodeAnalysisResult
{
    /// <summary>
    /// Path to the analyzed file
    /// </summary>
    public string FilePath { get; set; }

    /// <summary>
    /// Whether the file needs improvement
    /// </summary>
    public bool NeedsImprovement { get; set; }

    /// <summary>
    /// List of issues found in the file
    /// </summary>
    public List<CodeIssue> Issues { get; set; } = new();

    /// <summary>
    /// List of metrics calculated for the file
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();

    /// <summary>
    /// Additional information about the file
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Represents an issue found in code
/// </summary>
public class CodeIssue
{
    /// <summary>
    /// Type of the issue
    /// </summary>
    public CodeIssueType Type { get; set; }

    /// <summary>
    /// Severity of the issue
    /// </summary>
    public IssueSeverity Severity { get; set; }

    /// <summary>
    /// Description of the issue
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// Line number where the issue was found
    /// </summary>
    public int LineNumber { get; set; }

    /// <summary>
    /// Column number where the issue was found
    /// </summary>
    public int ColumnNumber { get; set; }

    /// <summary>
    /// Length of the code segment with the issue
    /// </summary>
    public int Length { get; set; }

    /// <summary>
    /// Code segment with the issue
    /// </summary>
    public string CodeSegment { get; set; }

    /// <summary>
    /// Suggested fix for the issue
    /// </summary>
    public string SuggestedFix { get; set; }
}

/// <summary>
/// Type of code issue
/// </summary>
public enum CodeIssueType
{
    /// <summary>
    /// Style issue (formatting, naming, etc.)
    /// </summary>
    Style,

    /// <summary>
    /// Performance issue
    /// </summary>
    Performance,

    /// <summary>
    /// Security issue
    /// </summary>
    Security,

    /// <summary>
    /// Maintainability issue
    /// </summary>
    Maintainability,

    /// <summary>
    /// Reliability issue
    /// </summary>
    Reliability,

    /// <summary>
    /// Documentation issue
    /// </summary>
    Documentation,

    /// <summary>
    /// Code duplication issue
    /// </summary>
    Duplication,

    /// <summary>
    /// Complexity issue
    /// </summary>
    Complexity,

    /// <summary>
    /// Design issue
    /// </summary>
    Design,

    /// <summary>
    /// Functional issue
    /// </summary>
    Functional
}

/// <summary>
/// Severity of a code issue
/// </summary>
public enum IssueSeverity
{
    /// <summary>
    /// Informational issue
    /// </summary>
    Info,

    /// <summary>
    /// Warning issue
    /// </summary>
    Warning,

    /// <summary>
    /// Error issue
    /// </summary>
    Error,

    /// <summary>
    /// Critical issue
    /// </summary>
    Critical
}