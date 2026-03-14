# Script to update the TreeOfThoughtExtensions.cs file

$content = @"
using Microsoft.Extensions.DependencyInjection;
using TarsEngine.Services.CodeAnalysis;
using TarsEngine.Services.Compilation;
using TarsEngine.Services.TreeOfThought;
using TarsCli.Commands;

namespace TarsEngine.Extensions
{
    /// <summary>
    /// Extension methods for registering Tree-of-Thought services.
    /// </summary>
    public static class TreeOfThoughtExtensions
    {
        /// <summary>
        /// Adds Tree-of-Thought services to the service collection.
        /// </summary>
        /// <param name="services">The service collection.</param>
        /// <returns>The service collection.</returns>
        public static IServiceCollection AddTreeOfThoughtServices(this IServiceCollection services)
        {
            // Add the F# script executor
            services.AddSingleton<FSharpScriptExecutor>();

            // Add the code analyzer
            services.AddSingleton<ICodeAnalyzer, BasicCodeAnalyzer>();

            // Add the Tree-of-Thought services
            services.AddSingleton<SimpleTreeOfThoughtService>();
            services.AddSingleton<EnhancedTreeOfThoughtService>();
            
            // Add the Tree-of-Thought commands
            services.AddSingleton<SimpleTreeOfThoughtCommand>();
            services.AddSingleton<TreeOfThoughtAutoImprovementCommand>();
            services.AddSingleton<EnhancedTreeOfThoughtCommand>();
            
            return services;
        }
    }
}
"@

Set-Content -Path "TarsEngine\Extensions\TreeOfThoughtExtensions.cs" -Value $content
