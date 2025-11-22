using System.Text.Json;

namespace TarsCli.Models
{
    /// <summary>
    /// Represents an MCP action
    /// </summary>
    public class McpAction
    {
        /// <summary>
        /// The type of action
        /// </summary>
        public string ActionType { get; set; } = string.Empty;

        /// <summary>
        /// The parameters for the action
        /// </summary>
        public JsonElement Parameters { get; set; }

        /// <summary>
        /// The ID of the action
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// The timestamp of the action
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }
}
