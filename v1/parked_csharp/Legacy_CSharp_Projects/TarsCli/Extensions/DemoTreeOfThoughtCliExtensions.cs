using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Commands;

namespace TarsCli.Extensions
{
    /// <summary>
    /// Extension methods for registering Demo Tree-of-Thought CLI commands.
    /// </summary>
    public static class DemoTreeOfThoughtCliExtensions
    {
        /// <summary>
        /// Adds the Demo Tree-of-Thought command to the root command.
        /// </summary>
        /// <param name="rootCommand">The root command.</param>
        /// <param name="serviceProvider">The service provider.</param>
        /// <returns>The root command.</returns>
        public static RootCommand AddDemoTreeOfThoughtCommand(this RootCommand rootCommand, IServiceProvider serviceProvider)
        {
            // Add the Demo Tree-of-Thought command
            rootCommand.AddCommand(serviceProvider.GetRequiredService<DemoAutoImprovementCommand>());
            
            return rootCommand;
        }
    }
}
