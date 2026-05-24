using Microsoft.Extensions.DependencyInjection;
using TarsEngine.Services.AutoImprovement;
using TarsEngine.Services.Compilation;
using TarsEngine.Services.TreeOfThought;
using TarsCli.Commands;

namespace TarsEngine.Extensions
{
    /// <summary>
    /// Extension methods for registering Metascript Tree-of-Thought services.
    /// </summary>
    public static class MetascriptTreeOfThoughtExtensions
    {
        /// <summary>
        /// Adds Metascript Tree-of-Thought services to the service collection.
        /// </summary>
        /// <param name="services">The service collection.</param>
        /// <returns>The service collection.</returns>
        public static IServiceCollection AddMetascriptTreeOfThoughtServices(this IServiceCollection services)
        {
            // Add the F# script executor
            services.AddSingleton<FSharpScriptExecutor>();
            
            // Add the Tree-of-Thought services
            services.AddSingleton<MetascriptTreeOfThoughtService>();
            
            // Add the Tree-of-Thought integration
            services.AddSingleton<MetascriptTreeOfThoughtIntegration>();
            
            // Add the Tree-of-Thought commands
            services.AddSingleton<MetascriptTreeOfThoughtCommand>();
            services.AddSingleton<MetascriptAutoImprovementCommand>();
            
            return services;
        }
    }
}
