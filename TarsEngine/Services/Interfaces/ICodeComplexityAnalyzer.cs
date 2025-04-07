using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models.Metrics;

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
    /// <returns>Complexity metrics for the file</returns>
    Task<List<ComplexityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes Halstead complexity of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>Complexity metrics for the file</returns>
    Task<List<ComplexityMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes all complexity metrics of a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="language">Programming language</param>
    /// <returns>All complexity metrics for the file</returns>
    Task<List<ComplexityMetric>> AnalyzeAllComplexityMetricsAsync(string filePath, string language);
    
    /// <summary>
    /// Analyzes complexity metrics of a project
    /// </summary>
    /// <param name="projectPath">Path to the project</param>
    /// <returns>Complexity metrics for the project</returns>
    Task<List<ComplexityMetric>> AnalyzeProjectComplexityAsync(string projectPath);
    
    /// <summary>
    /// Gets complexity thresholds for a specific language and complexity type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="complexityType">Type of complexity</param>
    /// <returns>Threshold values for the complexity type</returns>
    Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, ComplexityType complexityType);
    
    /// <summary>
    /// Sets complexity threshold for a specific language, complexity type, and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="complexityType">Type of complexity</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    Task<bool> SetComplexityThresholdAsync(string language, ComplexityType complexityType, string targetType, double threshold);
}
