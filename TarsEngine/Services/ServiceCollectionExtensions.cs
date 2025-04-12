using Microsoft.Extensions.DependencyInjection;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services
{
    /// <summary>
    /// Extension methods for IServiceCollection
    /// </summary>
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Adds code analysis services to the service collection
        /// </summary>
        /// <param name="services">The service collection</param>
        /// <returns>The service collection</returns>
        public static IServiceCollection AddCodeAnalysisServices(this IServiceCollection services)
        {
            // Register language detector
            services.AddSingleton<LanguageDetector>();

            // Register structure extractors
            services.AddSingleton<ICodeStructureExtractor, CSharpStructureExtractor>();
            services.AddSingleton<ICodeStructureExtractor, FSharpStructureExtractor>();

            // Register metrics calculator
            services.AddSingleton<IMetricsCalculator, MetricsCalculator>();

            // Register analyzers
            services.AddSingleton<SecurityAnalyzer>();
            services.AddSingleton<StyleAnalyzer>();

            // Register language analyzers
            services.AddSingleton<ILanguageAnalyzer, CSharpAnalyzer>();

            // Register code analysis service
            services.AddSingleton<ICodeAnalysisService, CodeAnalysisService>();

            return services;
        }
    }
}
