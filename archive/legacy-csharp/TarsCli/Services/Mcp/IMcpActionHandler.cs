using System.Threading.Tasks;
using TarsCli.Models;

namespace TarsCli.Services.Mcp
{
    /// <summary>
    /// Interface for MCP action handlers
    /// </summary>
    public interface IMcpActionHandler
    {
        /// <summary>
        /// Gets the action type that this handler can process
        /// </summary>
        string ActionType { get; }

        /// <summary>
        /// Handles an MCP action
        /// </summary>
        /// <param name="action">The action to handle</param>
        /// <returns>The result of handling the action</returns>
        Task<McpActionResult> HandleActionAsync(McpAction action);
    }
}