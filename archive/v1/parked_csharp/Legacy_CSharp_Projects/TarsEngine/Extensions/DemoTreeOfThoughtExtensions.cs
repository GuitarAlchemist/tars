using Microsoft.Extensions.DependencyInjection;
using TarsEngine.Services.TreeOfThought;
using TarsCli.Commands;

namespace TarsEngine.Extensions
{
    /// <summary>
    /// Extension methods for registering Demo Tree-of-Thought services.
    /// </summary>
    public static class DemoTreeOfThoughtExtensions
    {
        /// <summary>
        /// Adds Demo Tree-of-Thought services to the service collection.
        /// </summary>
        /// <param name="services">The service collection.</param>
        /// <returns>The service collection.</returns>
        public static IServiceCollection AddDemoTreeOfThoughtServices(this IServiceCollection services)
        {
            // Add the Tree-of-Thought services
            services.AddSingleton<DemoTreeOfThoughtService>();
            
            // Add the Tree-of-Thought commands
            services.AddSingleton<DemoAutoImprovementCommand>();
            
            return services;
        }
    }
}
