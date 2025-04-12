using System.Collections.Generic;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces
{
    /// <summary>
    /// Interface for calculating code metrics
    /// </summary>
    public interface IMetricsCalculator
    {
        /// <summary>
        /// Gets the language supported by this calculator
        /// </summary>
        string Language { get; }

        /// <summary>
        /// Calculates metrics for the provided code content
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <param name="structures">The extracted code structures</param>
        /// <param name="analyzeComplexity">Whether to analyze complexity</param>
        /// <param name="analyzeMaintainability">Whether to analyze maintainability</param>
        /// <returns>A list of calculated metrics</returns>
        List<CodeMetric> CalculateMetrics(string content, List<CodeStructure> structures, bool analyzeComplexity, bool analyzeMaintainability);

        /// <summary>
        /// Calculates the cyclomatic complexity of a method
        /// </summary>
        /// <param name="methodContent">The method content</param>
        /// <returns>The cyclomatic complexity value</returns>
        int CalculateCyclomaticComplexity(string methodContent);

        /// <summary>
        /// Calculates the maintainability index of the code
        /// </summary>
        /// <param name="halsteadVolume">The Halstead volume</param>
        /// <param name="cyclomaticComplexity">The cyclomatic complexity</param>
        /// <param name="linesOfCode">The lines of code</param>
        /// <returns>The maintainability index value</returns>
        double CalculateMaintainabilityIndex(double halsteadVolume, double cyclomaticComplexity, int linesOfCode);

        /// <summary>
        /// Gets the available metric types for this calculator
        /// </summary>
        /// <returns>A dictionary of metric types and their descriptions</returns>
        Dictionary<MetricType, string> GetAvailableMetricTypes();
    }
}
