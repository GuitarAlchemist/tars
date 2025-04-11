using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the code quality service
/// </summary>
public interface ICodeQualityService
{
    /// <summary>
    /// Analyzes code quality for a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="language">Programming language</param>
    /// <returns>Code quality result</returns>
    Task<CodeQualityResult> AnalyzeCodeQualityAsync(string filePath, string language);

    /// <summary>
    /// Analyzes code quality for a specific project
    /// </summary>
    /// <param name="projectPath">Path to the project to analyze</param>
    /// <returns>Code quality result</returns>
    Task<CodeQualityResult> AnalyzeProjectQualityAsync(string projectPath);

    /// <summary>
    /// Analyzes code quality for a specific solution
    /// </summary>
    /// <param name="solutionPath">Path to the solution to analyze</param>
    /// <returns>Code quality result</returns>
    Task<CodeQualityResult> AnalyzeSolutionQualityAsync(string solutionPath);

    /// <summary>
    /// Tracks quality scores over time
    /// </summary>
    /// <param name="projectPath">Path to the project to track</param>
    /// <returns>Quality trend result</returns>
    Task<QualityTrendResult> TrackQualityScoresAsync(string projectPath);

    /// <summary>
    /// Suggests quality improvements
    /// </summary>
    /// <param name="qualityResult">Code quality result</param>
    /// <returns>List of suggested improvements</returns>
    Task<List<QualityImprovement>> SuggestQualityImprovementsAsync(CodeQualityResult qualityResult);
}

/// <summary>
/// Represents a code quality result
/// </summary>
public class CodeQualityResult
{
    /// <summary>
    /// Overall quality score (0-100)
    /// </summary>
    public float OverallScore { get; set; }

    /// <summary>
    /// Maintainability score (0-100)
    /// </summary>
    public float MaintainabilityScore { get; set; }

    /// <summary>
    /// Reliability score (0-100)
    /// </summary>
    public float ReliabilityScore { get; set; }

    /// <summary>
    /// Security score (0-100)
    /// </summary>
    public float SecurityScore { get; set; }

    /// <summary>
    /// Performance score (0-100)
    /// </summary>
    public float PerformanceScore { get; set; }

    /// <summary>
    /// List of quality issues
    /// </summary>
    public List<QualityIssue> Issues { get; set; } = new();

    /// <summary>
    /// Complexity metrics
    /// </summary>
    public ComplexityMetrics ComplexityMetrics { get; set; } = new();

    /// <summary>
    /// Readability metrics
    /// </summary>
    public ReadabilityMetrics ReadabilityMetrics { get; set; } = new();

    /// <summary>
    /// Code duplication metrics
    /// </summary>
    public DuplicationMetrics DuplicationMetrics { get; set; } = new();
}

/// <summary>
/// Represents a quality issue
/// </summary>
public class QualityIssue
{
    /// <summary>
    /// Description of the issue
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Severity of the issue
    /// </summary>
    public IssueSeverity Severity { get; set; } = IssueSeverity.Warning;

    /// <summary>
    /// Category of the issue
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Location of the issue
    /// </summary>
    public string Location { get; set; } = string.Empty;

    /// <summary>
    /// Suggested fix for the issue
    /// </summary>
    public string? SuggestedFix { get; set; }

    /// <summary>
    /// Impact on quality score
    /// </summary>
    public float ScoreImpact { get; set; }
}

/// <summary>
/// Represents complexity metrics
/// </summary>
public class ComplexityMetrics
{
    /// <summary>
    /// Average cyclomatic complexity
    /// </summary>
    public float AverageCyclomaticComplexity { get; set; }

    /// <summary>
    /// Maximum cyclomatic complexity
    /// </summary>
    public float MaxCyclomaticComplexity { get; set; }

    /// <summary>
    /// Average cognitive complexity
    /// </summary>
    public float AverageCognitiveComplexity { get; set; }

    /// <summary>
    /// Maximum cognitive complexity
    /// </summary>
    public float MaxCognitiveComplexity { get; set; }

    /// <summary>
    /// Average method length
    /// </summary>
    public float AverageMethodLength { get; set; }

    /// <summary>
    /// Maximum method length
    /// </summary>
    public int MaxMethodLength { get; set; }

    /// <summary>
    /// Average class length
    /// </summary>
    public float AverageClassLength { get; set; }

    /// <summary>
    /// Maximum class length
    /// </summary>
    public int MaxClassLength { get; set; }

    /// <summary>
    /// List of complex methods
    /// </summary>
    public List<ComplexMethod> ComplexMethods { get; set; } = new();
}

/// <summary>
/// Represents a complex method
/// </summary>
public class ComplexMethod
{
    /// <summary>
    /// Method name
    /// </summary>
    public string MethodName { get; set; } = string.Empty;

    /// <summary>
    /// Class name
    /// </summary>
    public string ClassName { get; set; } = string.Empty;

    /// <summary>
    /// File path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Line number
    /// </summary>
    public int LineNumber { get; set; }

    /// <summary>
    /// Cyclomatic complexity
    /// </summary>
    public int CyclomaticComplexity { get; set; }

    /// <summary>
    /// Cognitive complexity
    /// </summary>
    public int CognitiveComplexity { get; set; }

    /// <summary>
    /// Method length in lines
    /// </summary>
    public int MethodLength { get; set; }
}

/// <summary>
/// Represents readability metrics
/// </summary>
public class ReadabilityMetrics
{
    /// <summary>
    /// Average identifier length
    /// </summary>
    public float AverageIdentifierLength { get; set; }

    /// <summary>
    /// Comment density (comments per line of code)
    /// </summary>
    public float CommentDensity { get; set; }

    /// <summary>
    /// Documentation coverage percentage
    /// </summary>
    public float DocumentationCoverage { get; set; }

    /// <summary>
    /// Average method parameter count
    /// </summary>
    public float AverageParameterCount { get; set; }

    /// <summary>
    /// Maximum method parameter count
    /// </summary>
    public int MaxParameterCount { get; set; }

    /// <summary>
    /// List of readability issues
    /// </summary>
    public List<ReadabilityIssue> ReadabilityIssues { get; set; } = new();
}

/// <summary>
/// Represents a readability issue
/// </summary>
public class ReadabilityIssue
{
    /// <summary>
    /// Description of the issue
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Location of the issue
    /// </summary>
    public string Location { get; set; } = string.Empty;

    /// <summary>
    /// Suggested fix for the issue
    /// </summary>
    public string? SuggestedFix { get; set; }

    /// <summary>
    /// Impact on readability score
    /// </summary>
    public float ScoreImpact { get; set; }
}

/// <summary>
/// Represents duplication metrics
/// </summary>
public class DuplicationMetrics
{
    /// <summary>
    /// Duplication percentage
    /// </summary>
    public float DuplicationPercentage { get; set; }

    /// <summary>
    /// Number of duplicated blocks
    /// </summary>
    public int DuplicatedBlocks { get; set; }

    /// <summary>
    /// Number of duplicated lines
    /// </summary>
    public int DuplicatedLines { get; set; }

    /// <summary>
    /// List of duplicated blocks
    /// </summary>
    public List<DuplicatedBlock> DuplicatedBlocksList { get; set; } = new();
}

/// <summary>
/// Represents a duplicated block
/// </summary>
public class DuplicatedBlock
{
    /// <summary>
    /// Source file path
    /// </summary>
    public string SourceFilePath { get; set; } = string.Empty;

    /// <summary>
    /// Source start line
    /// </summary>
    public int SourceStartLine { get; set; }

    /// <summary>
    /// Source end line
    /// </summary>
    public int SourceEndLine { get; set; }

    /// <summary>
    /// Target file path
    /// </summary>
    public string TargetFilePath { get; set; } = string.Empty;

    /// <summary>
    /// Target start line
    /// </summary>
    public int TargetStartLine { get; set; }

    /// <summary>
    /// Target end line
    /// </summary>
    public int TargetEndLine { get; set; }

    /// <summary>
    /// Duplicated code
    /// </summary>
    public string DuplicatedCode { get; set; } = string.Empty;

    /// <summary>
    /// Similarity percentage
    /// </summary>
    public double SimilarityPercentage { get; set; }

    /// <summary>
    /// File path (for backward compatibility)
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Start line (for backward compatibility)
    /// </summary>
    public int StartLine { get; set; }

    /// <summary>
    /// End line (for backward compatibility)
    /// </summary>
    public int EndLine { get; set; }

    /// <summary>
    /// Duplicated content (for backward compatibility)
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// List of duplicate locations
    /// </summary>
    public List<DuplicateLocation> DuplicateLocations { get; set; } = new();
}

// This class has been moved to TarsEngine.Services.Interfaces.DuplicateLocation

/// <summary>
/// Represents a quality trend result
/// </summary>
public class QualityTrendResult
{
    /// <summary>
    /// List of quality snapshots
    /// </summary>
    public List<QualitySnapshot> Snapshots { get; set; } = new();

    /// <summary>
    /// Overall trend direction
    /// </summary>
    public TrendDirection OverallTrend { get; set; } = TrendDirection.Stable;

    /// <summary>
    /// Maintainability trend direction
    /// </summary>
    public TrendDirection MaintainabilityTrend { get; set; } = TrendDirection.Stable;

    /// <summary>
    /// Reliability trend direction
    /// </summary>
    public TrendDirection ReliabilityTrend { get; set; } = TrendDirection.Stable;

    /// <summary>
    /// Security trend direction
    /// </summary>
    public TrendDirection SecurityTrend { get; set; } = TrendDirection.Stable;

    /// <summary>
    /// Performance trend direction
    /// </summary>
    public TrendDirection PerformanceTrend { get; set; } = TrendDirection.Stable;
}

/// <summary>
/// Represents a quality snapshot
/// </summary>
public class QualitySnapshot
{
    /// <summary>
    /// Timestamp of the snapshot
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Overall quality score
    /// </summary>
    public float OverallScore { get; set; }

    /// <summary>
    /// Maintainability score
    /// </summary>
    public float MaintainabilityScore { get; set; }

    /// <summary>
    /// Reliability score
    /// </summary>
    public float ReliabilityScore { get; set; }

    /// <summary>
    /// Security score
    /// </summary>
    public float SecurityScore { get; set; }

    /// <summary>
    /// Performance score
    /// </summary>
    public float PerformanceScore { get; set; }

    /// <summary>
    /// Number of issues
    /// </summary>
    public int IssueCount { get; set; }

    /// <summary>
    /// Commit hash (if available)
    /// </summary>
    public string? CommitHash { get; set; }
}

/// <summary>
/// Represents a trend direction
/// </summary>
public enum TrendDirection
{
    Improving,
    Stable,
    Declining
}

/// <summary>
/// Represents a quality improvement
/// </summary>
public class QualityImprovement
{
    /// <summary>
    /// Description of the improvement
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Category of the improvement
    /// </summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Priority of the improvement
    /// </summary>
    public ImprovementPriority Priority { get; set; } = ImprovementPriority.Medium;

    /// <summary>
    /// Estimated effort to implement (in hours)
    /// </summary>
    public float EstimatedEffort { get; set; }

    /// <summary>
    /// Estimated impact on quality score
    /// </summary>
    public float EstimatedScoreImpact { get; set; }

    /// <summary>
    /// List of affected files
    /// </summary>
    public List<string> AffectedFiles { get; set; } = new();

    /// <summary>
    /// Implementation steps
    /// </summary>
    public List<string> ImplementationSteps { get; set; } = new();
}

/// <summary>
/// Priority of an improvement
/// </summary>
public enum ImprovementPriority
{
    Low,
    Medium,
    High,
    Critical
}
