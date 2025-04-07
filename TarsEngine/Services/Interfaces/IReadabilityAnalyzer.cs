using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for readability analyzer
/// </summary>
public interface IReadabilityAnalyzer
{
    /// <summary>
    /// Analyzes identifier quality of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Readability metrics for the file</returns>
    Task<List<ReadabilityMetric>> AnalyzeIdentifierQualityAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes comment quality of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Readability metrics for the file</returns>
    Task<List<ReadabilityMetric>> AnalyzeCommentQualityAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes code structure of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Readability metrics for the file</returns>
    Task<List<ReadabilityMetric>> AnalyzeCodeStructureAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes overall readability of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Readability metrics for the file</returns>
    Task<List<ReadabilityMetric>> AnalyzeOverallReadabilityAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes all readability metrics of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>All readability metrics for the file</returns>
    Task<List<ReadabilityMetric>> AnalyzeAllReadabilityMetricsAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes readability metrics of a project
    /// </summary>
    /// <param name="projectPath">Path to the project</param>
    /// <returns>Readability metrics for the project</returns>
    Task<List<ReadabilityMetric>> AnalyzeProjectReadabilityAsync(string projectPath);
    
    /// <summary>
    /// Gets readability thresholds for a specific language and readability type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="readabilityType">Type of readability</param>
    /// <returns>Threshold values for the readability type</returns>
    Task<Dictionary<string, double>> GetReadabilityThresholdsAsync(string language, ReadabilityType readabilityType);
    
    /// <summary>
    /// Sets readability threshold for a specific language, readability type, and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="readabilityType">Type of readability</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    Task<bool> SetReadabilityThresholdAsync(string language, ReadabilityType readabilityType, string targetType, double threshold);
}
