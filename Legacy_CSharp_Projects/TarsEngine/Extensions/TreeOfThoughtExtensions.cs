using Microsoft.Extensions.DependencyInjection;
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
            
            // Add the Tree-of-Thought services
            services.AddSingleton<SimpleTreeOfThoughtService>();
            
            // Add the Tree-of-Thought commands
            services.AddSingleton<SimpleTreeOfThoughtCommand>();
            
            return services;
        }
    }
}
