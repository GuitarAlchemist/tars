using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models.Metrics;
using UnifiedComplexityType = TarsEngine.Models.Unified.ComplexityTypeUnified;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for code complexity analyzer
/// </summary>
public interface ICodeComplexityAnalyzer
{
    /// <summary>
    /// Analyzes cyclomatic complexity of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Complexity metrics for the file</returns>
    Task<List<ComplexityMetric>> AnalyzeCyclomaticComplexityAsync(string filePath, string language);

    /// <summary>
    /// Analyzes cognitive complexity of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Complexity metrics for the file</returns>
    Task<List<ComplexityMetric>> AnalyzeCognitiveComplexityAsync(string filePath, string language);

    /// <summary>
    /// Analyzes maintainability index of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Maintainability metrics for the file</returns>
    Task<List<MaintainabilityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language);

    /// <summary>
    /// Analyzes Halstead complexity of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Halstead complexity metrics for the file</returns>
    Task<List<HalsteadMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language);

    /// <summary>
    /// Analyzes readability of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <param name="readabilityType">Type of readability to analyze</param>
    /// <returns>Readability metrics for the file</returns>
    Task<List<ReadabilityMetric>> AnalyzeReadabilityAsync(string filePath, string language, ReadabilityType readabilityType);

    /// <summary>
    /// Analyzes all complexity metrics of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>All complexity metrics for the file</returns>
    Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics, List<ReadabilityMetric> ReadabilityMetrics)> AnalyzeAllComplexityMetricsAsync(string filePath, string language);

    /// <summary>
    /// Analyzes complexity metrics of a project
    /// </summary>
    /// <param name="projectPath">Path to the project</param>
    /// <returns>All complexity metrics for the project</returns>
    Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics, List<ReadabilityMetric> ReadabilityMetrics)> AnalyzeProjectComplexityAsync(string projectPath);

    /// <summary>
    /// Gets complexity thresholds for a specific language and complexity type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="complexityType">Type of complexity</param>
    /// <returns>Threshold values for the complexity type</returns>
    Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, UnifiedComplexityType complexityType);

    /// <summary>
    /// Gets Halstead complexity thresholds for a specific language and Halstead type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="halsteadType">Type of Halstead metric</param>
    /// <returns>Threshold values for the Halstead type</returns>
    Task<Dictionary<string, double>> GetHalsteadThresholdsAsync(string language, HalsteadType halsteadType);

    /// <summary>
    /// Gets maintainability index thresholds for a specific language
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <returns>Threshold values for maintainability index</returns>
    Task<Dictionary<string, double>> GetMaintainabilityThresholdsAsync(string language);

    /// <summary>
    /// Sets complexity threshold for a specific language, complexity type, and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="complexityType">Type of complexity</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    Task<bool> SetComplexityThresholdAsync(string language, UnifiedComplexityType complexityType, string targetType, double threshold);

    /// <summary>
    /// Sets Halstead complexity threshold for a specific language, Halstead type, and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="halsteadType">Type of Halstead metric</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    Task<bool> SetHalsteadThresholdAsync(string language, HalsteadType halsteadType, string targetType, double threshold);

    /// <summary>
    /// Sets maintainability index threshold for a specific language and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    Task<bool> SetMaintainabilityThresholdAsync(string language, string targetType, double threshold);
}
