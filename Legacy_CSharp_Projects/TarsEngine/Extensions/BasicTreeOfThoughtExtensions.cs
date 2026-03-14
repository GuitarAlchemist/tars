using Microsoft.Extensions.DependencyInjection;
using TarsEngine.Services.TreeOfThought;
using TarsCli.Commands;

namespace TarsEngine.Extensions
{
    /// <summary>
    /// Extension methods for registering Basic Tree-of-Thought services.
    /// </summary>
    public static class BasicTreeOfThoughtExtensions
    {
        /// <summary>
        /// Adds Basic Tree-of-Thought services to the service collection.
        /// </summary>
        /// <param name="services">The service collection.</param>
        /// <returns>The service collection.</returns>
        public static IServiceCollection AddBasicTreeOfThoughtServices(this IServiceCollection services)
        {
            // Add the Tree-of-Thought services
            services.AddSingleton<BasicTreeOfThoughtService>();
            
            // Add the Tree-of-Thought commands
            services.AddSingleton<BasicAutoImprovementCommand>();
            
            return services;
        }
    }
}
