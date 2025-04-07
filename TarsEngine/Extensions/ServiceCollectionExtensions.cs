using Microsoft.Extensions.DependencyInjection;
using TarsEngine.Data;
using TarsEngine.Intelligence.Measurement;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Extensions;

/// <summary>
/// Extension methods for <see cref="IServiceCollection"/>
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds intelligence progression measurement services
    /// </summary>
    /// <param name="services">The service collection</param>
    /// <returns>The service collection</returns>
    public static IServiceCollection AddIntelligenceProgressionMeasurement(this IServiceCollection services)
    {
        // Add repositories
        services.AddSingleton<MetricsRepository>();

        // Add measurement components
        services.AddSingleton<MetricsCollector>();
        services.AddSingleton<LearningCurveAnalyzer>();
        services.AddSingleton<ModificationAnalyzer>();
        services.AddSingleton<ProgressionVisualizer>();

        // Add code complexity analyzers
        services.AddSingleton<CSharpComplexityAnalyzer>();
        services.AddSingleton<FSharpComplexityAnalyzer>();
        services.AddSingleton<ICodeComplexityAnalyzer, CodeComplexityAnalyzerService>();

        // Add the main system
        services.AddSingleton<IntelligenceProgressionSystem>();

        return services;
    }
}
