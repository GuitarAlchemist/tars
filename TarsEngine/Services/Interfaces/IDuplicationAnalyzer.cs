using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for code duplication analyzer
/// </summary>
public interface IDuplicationAnalyzer
{
    /// <summary>
    /// Analyzes token-based duplication in a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Duplication metrics for the file</returns>
    Task<List<DuplicationMetric>> AnalyzeTokenBasedDuplicationAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes semantic duplication in a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Duplication metrics for the file</returns>
    Task<List<DuplicationMetric>> AnalyzeSemanticDuplicationAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes all duplication types in a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>All duplication metrics for the file</returns>
    Task<List<DuplicationMetric>> AnalyzeAllDuplicationMetricsAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes duplication in a project
    /// </summary>
    /// <param name="projectPath">Path to the project</param>
    /// <returns>Duplication metrics for the project</returns>
    Task<List<DuplicationMetric>> AnalyzeProjectDuplicationAsync(string projectPath);
    
    /// <summary>
    /// Gets duplication thresholds for a specific language and duplication type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="duplicationType">Type of duplication</param>
    /// <returns>Threshold values for the duplication type</returns>
    Task<Dictionary<string, double>> GetDuplicationThresholdsAsync(string language, DuplicationType duplicationType);
    
    /// <summary>
    /// Sets duplication threshold for a specific language, duplication type, and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="duplicationType">Type of duplication</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    Task<bool> SetDuplicationThresholdAsync(string language, DuplicationType duplicationType, string targetType, double threshold);
    
    /// <summary>
    /// Visualizes duplication in a file or project
    /// </summary>
    /// <param name="path">Path to the file or project</param>
    /// <param name="language">Programming language</param>
    /// <param name="outputPath">Path to save the visualization</param>
    /// <returns>True if visualization was created successfully</returns>
    Task<bool> VisualizeDuplicationAsync(string path, string language, string outputPath);
}
