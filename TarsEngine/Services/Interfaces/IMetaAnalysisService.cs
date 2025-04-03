using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Services.Models;

namespace TarsEngine.Services.Interfaces
{
    /// <summary>
    /// Interface for the meta-analysis service that enables TARS to analyze its own codebase
    /// and make strategic decisions about what to improve
    /// </summary>
    public interface IMetaAnalysisService
    {
        /// <summary>
        /// Analyzes the entire codebase to identify components that need improvement
        /// </summary>
        /// <param name="rootPath">Root path of the codebase</param>
        /// <returns>A meta-analysis result containing information about components that need improvement</returns>
        Task<MetaAnalysisResult> AnalyzeCodebaseAsync(string rootPath);

        /// <summary>
        /// Identifies the most critical components to improve based on various factors
        /// </summary>
        /// <param name="metaAnalysis">The meta-analysis result</param>
        /// <param name="maxComponents">Maximum number of components to return</param>
        /// <returns>A list of components to improve, ordered by priority</returns>
        Task<List<ComponentToImprove>> IdentifyCriticalComponentsAsync(MetaAnalysisResult metaAnalysis, int maxComponents = 5);

        /// <summary>
        /// Determines the best improvement strategy for a component
        /// </summary>
        /// <param name="component">The component to improve</param>
        /// <returns>The recommended improvement strategy</returns>
        Task<ImprovementStrategy> DetermineImprovementStrategyAsync(ComponentToImprove component);

        /// <summary>
        /// Evaluates the impact of improvements made to a component
        /// </summary>
        /// <param name="component">The component that was improved</param>
        /// <param name="beforeMetrics">Metrics before the improvement</param>
        /// <param name="afterMetrics">Metrics after the improvement</param>
        /// <returns>An impact assessment of the improvements</returns>
        Task<ImpactAssessment> EvaluateImprovementImpactAsync(
            ComponentToImprove component, 
            ComponentMetrics beforeMetrics, 
            ComponentMetrics afterMetrics);
    }
}
