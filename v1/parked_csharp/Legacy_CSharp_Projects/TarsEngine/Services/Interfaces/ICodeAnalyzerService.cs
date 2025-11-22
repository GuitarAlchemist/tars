using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the code analyzer service
/// </summary>
public interface ICodeAnalyzerService
{
    /// <summary>
    /// Analyzes a file
    /// </summary>
    /// <param name="filePath">The path to the file to analyze</param>
    /// <param name="options">Optional analysis options</param>
    /// <returns>The analysis result</returns>
    Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath, Dictionary<string, string>? options = null);

    /// <summary>
    /// Analyzes a directory
    /// </summary>
    /// <param name="directoryPath">The path to the directory to analyze</param>
    /// <param name="recursive">Whether to analyze subdirectories</param>
    /// <param name="filePattern">The pattern to match files to analyze</param>
    /// <param name="options">Optional analysis options</param>
    /// <returns>The analysis results</returns>
    Task<List<CodeAnalysisResult>> AnalyzeDirectoryAsync(string directoryPath, bool recursive = true, string filePattern = "*.cs;*.fs", Dictionary<string, string>? options = null);

    /// <summary>
    /// Analyzes code content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="language">The programming language of the code</param>
    /// <param name="options">Optional analysis options</param>
    /// <returns>The analysis result</returns>
    Task<CodeAnalysisResult> AnalyzeContentAsync(string content, string language, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the supported languages for analysis
    /// </summary>
    /// <returns>The list of supported languages</returns>
    Task<List<string>> GetSupportedLanguagesAsync();

    /// <summary>
    /// Gets the available analysis options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    Task<Dictionary<string, string>> GetAvailableOptionsAsync();

    /// <summary>
    /// Gets the available issue types
    /// </summary>
    /// <returns>The dictionary of available issue types and their descriptions</returns>
    Task<Dictionary<CodeIssueType, string>> GetAvailableIssueTypesAsync();

    /// <summary>
    /// Gets the available metric types
    /// </summary>
    /// <returns>The dictionary of available metric types and their descriptions</returns>
    Task<Dictionary<MetricType, string>> GetAvailableMetricTypesAsync();

    /// <summary>
    /// Gets the thresholds for metrics
    /// </summary>
    /// <returns>The dictionary of metric types and their thresholds</returns>
    Task<Dictionary<MetricType, (double Good, double Acceptable, double Poor)>> GetMetricThresholdsAsync();

    /// <summary>
    /// Sets the thresholds for a metric
    /// </summary>
    /// <param name="metricType">The metric type</param>
    /// <param name="goodThreshold">The threshold for good values</param>
    /// <param name="acceptableThreshold">The threshold for acceptable values</param>
    /// <param name="poorThreshold">The threshold for poor values</param>
    /// <returns>True if the thresholds were set successfully, false otherwise</returns>
    Task<bool> SetMetricThresholdsAsync(MetricType metricType, double goodThreshold, double acceptableThreshold, double poorThreshold);

    /// <summary>
    /// Gets the issues for a specific file
    /// </summary>
    /// <param name="filePath">The path to the file</param>
    /// <param name="issueTypes">The types of issues to get</param>
    /// <param name="minSeverity">The minimum severity of issues to get</param>
    /// <param name="options">Optional filtering options</param>
    /// <returns>The list of issues</returns>
    Task<List<CodeIssue>> GetIssuesForFileAsync(string filePath, List<CodeIssueType>? issueTypes = null, TarsEngine.Models.IssueSeverity minSeverity = TarsEngine.Models.IssueSeverity.Info, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the metrics for a specific file
    /// </summary>
    /// <param name="filePath">The path to the file</param>
    /// <param name="metricTypes">The types of metrics to get</param>
    /// <param name="scope">The scope of metrics to get</param>
    /// <param name="options">Optional filtering options</param>
    /// <returns>The list of metrics</returns>
    Task<List<CodeMetric>> GetMetricsForFileAsync(string filePath, List<MetricType>? metricTypes = null, MetricScope? scope = null, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the structures for a specific file
    /// </summary>
    /// <param name="filePath">The path to the file</param>
    /// <param name="structureTypes">The types of structures to get</param>
    /// <param name="options">Optional filtering options</param>
    /// <returns>The list of structures</returns>
    Task<List<CodeStructure>> GetStructuresForFileAsync(string filePath, List<StructureType>? structureTypes = null, Dictionary<string, string>? options = null);
}
