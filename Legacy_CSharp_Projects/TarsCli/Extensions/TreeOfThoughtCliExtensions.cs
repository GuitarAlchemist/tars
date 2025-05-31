using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Commands;

namespace TarsCli.Extensions
{
    /// <summary>
    /// Extension methods for registering Tree-of-Thought CLI commands.
    /// </summary>
    public static class TreeOfThoughtCliExtensions
    {
        /// <summary>
        /// Adds the Tree-of-Thought command to the root command.
        /// </summary>
        /// <param name="rootCommand">The root command.</param>
        /// <param name="serviceProvider">The service provider.</param>
        /// <returns>The root command.</returns>
        public static RootCommand AddTreeOfThoughtCommand(this RootCommand rootCommand, IServiceProvider serviceProvider)
        {
            // Add the Tree-of-Thought command
            rootCommand.AddCommand(serviceProvider.GetRequiredService<TreeOfThoughtAutoImprovementCommand>());
            
            return rootCommand;
        }
    }
}
