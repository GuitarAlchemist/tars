using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# complexity analysis functionality.
    /// </summary>
    public class ComplexityAnalysisAdapter
    {
        private readonly ILogger<ComplexityAnalysisAdapter> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="ComplexityAnalysisAdapter"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public ComplexityAnalysisAdapter(ILogger<ComplexityAnalysisAdapter> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Analyzes the cyclomatic complexity of a file.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="language">The language of the file.</param>
        /// <returns>The complexity metrics.</returns>
        public IReadOnlyList<ComplexityMetricAdapter> AnalyzeCyclomaticComplexity(string filePath, string language)
        {
            try
            {
                _logger.LogInformation("Analyzing cyclomatic complexity of {Language} file {FilePath}", language, filePath);

                var metrics = ComplexityAnalysis.analyzeCyclomaticComplexity(filePath, language);

                return metrics.Select(m => new ComplexityMetricAdapter(m)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing cyclomatic complexity of {Language} file {FilePath}", language, filePath);
                return Array.Empty<ComplexityMetricAdapter>();
            }
        }

        /// <summary>
        /// Analyzes the cognitive complexity of a file.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="language">The language of the file.</param>
        /// <returns>The complexity metrics.</returns>
        public IReadOnlyList<ComplexityMetricAdapter> AnalyzeCognitiveComplexity(string filePath, string language)
        {
            try
            {
                _logger.LogInformation("Analyzing cognitive complexity of {Language} file {FilePath}", language, filePath);

                var metrics = ComplexityAnalysis.analyzeCognitiveComplexity(filePath, language);

                return metrics.Select(m => new ComplexityMetricAdapter(m)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing cognitive complexity of {Language} file {FilePath}", language, filePath);
                return Array.Empty<ComplexityMetricAdapter>();
            }
        }

        /// <summary>
        /// Analyzes the maintainability index of a file.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="language">The language of the file.</param>
        /// <returns>The maintainability metrics.</returns>
        public IReadOnlyList<MaintainabilityMetricAdapter> AnalyzeMaintainabilityIndex(string filePath, string language)
        {
            try
            {
                _logger.LogInformation("Analyzing maintainability index of {Language} file {FilePath}", language, filePath);

                var metrics = ComplexityAnalysis.analyzeMaintainabilityIndex(filePath, language);

                return metrics.Select(m => new MaintainabilityMetricAdapter(m)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing maintainability index of {Language} file {FilePath}", language, filePath);
                return Array.Empty<MaintainabilityMetricAdapter>();
            }
        }

        /// <summary>
        /// Analyzes the Halstead complexity of a file.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="language">The language of the file.</param>
        /// <returns>The Halstead complexity metrics.</returns>
        public IReadOnlyList<HalsteadMetricAdapter> AnalyzeHalsteadComplexity(string filePath, string language)
        {
            try
            {
                _logger.LogInformation("Analyzing Halstead complexity of {Language} file {FilePath}", language, filePath);

                var metrics = ComplexityAnalysis.analyzeHalsteadComplexity(filePath, language);

                return metrics.Select(m => new HalsteadMetricAdapter(m)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing Halstead complexity of {Language} file {FilePath}", language, filePath);
                return Array.Empty<HalsteadMetricAdapter>();
            }
        }
    }
}
