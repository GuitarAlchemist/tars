using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the readability service
/// </summary>
public interface IReadabilityService
{
    /// <summary>
    /// Analyzes code readability for a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="language">Programming language</param>
    /// <returns>Readability analysis result</returns>
    Task<ReadabilityAnalysisResult> AnalyzeReadabilityAsync(string filePath, string language);

    /// <summary>
    /// Analyzes code readability for a specific project
    /// </summary>
    /// <param name="projectPath">Path to the project to analyze</param>
    /// <returns>Readability analysis result</returns>
    Task<ReadabilityAnalysisResult> AnalyzeProjectReadabilityAsync(string projectPath);

    /// <summary>
    /// Analyzes code readability for a specific solution
    /// </summary>
    /// <param name="solutionPath">Path to the solution to analyze</param>
    /// <returns>Readability analysis result</returns>
    Task<ReadabilityAnalysisResult> AnalyzeSolutionReadabilityAsync(string solutionPath);

    /// <summary>
    /// Identifies hard-to-read code
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="language">Programming language</param>
    /// <returns>List of readability issues</returns>
    Task<List<ReadabilityIssue>> IdentifyReadabilityIssuesAsync(string filePath, string language);

    /// <summary>
    /// Suggests improvements for readability issues
    /// </summary>
    /// <param name="readabilityIssue">Readability issue</param>
    /// <returns>List of suggested improvements</returns>
    Task<List<ReadabilityImprovement>> SuggestReadabilityImprovementsAsync(ReadabilityIssue readabilityIssue);
}

/// <summary>
/// Represents a readability analysis result
/// </summary>
public class ReadabilityAnalysisResult
{
    /// <summary>
    /// Overall readability score (0-100)
    /// </summary>
    public float OverallScore { get; set; }

    /// <summary>
    /// Naming convention score (0-100)
    /// </summary>
    public float NamingConventionScore { get; set; }

    /// <summary>
    /// Comment quality score (0-100)
    /// </summary>
    public float CommentQualityScore { get; set; }

    /// <summary>
    /// Code formatting score (0-100)
    /// </summary>
    public float CodeFormattingScore { get; set; }

    /// <summary>
    /// Documentation score (0-100)
    /// </summary>
    public float DocumentationScore { get; set; }

    /// <summary>
    /// List of readability issues
    /// </summary>
    public List<ReadabilityIssue> Issues { get; set; } = new();

    /// <summary>
    /// Readability metrics
    /// </summary>
    public ReadabilityMetrics Metrics { get; set; } = new();
}

/// <summary>
/// Represents a readability improvement
/// </summary>
public class ReadabilityImprovement
{
    /// <summary>
    /// Description of the improvement
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Improved code
    /// </summary>
    public string ImprovedCode { get; set; } = string.Empty;

    /// <summary>
    /// Readability improvement score
    /// </summary>
    public float ReadabilityScore { get; set; }

    /// <summary>
    /// Confidence in the improvement (0-1)
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Category of the improvement
    /// </summary>
    public ReadabilityImprovementCategory Category { get; set; } = ReadabilityImprovementCategory.Naming;
}

/// <summary>
/// Category of a readability improvement
/// </summary>
public enum ReadabilityImprovementCategory
{
    Naming,
    Comments,
    Formatting,
    Documentation,
    MethodStructure,
    ClassStructure,
    VariableUsage
}
