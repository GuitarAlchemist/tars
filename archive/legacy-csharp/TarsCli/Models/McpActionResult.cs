using System.Text.Json;

namespace TarsCli.Models
{
    /// <summary>
    /// Represents the result of an MCP action
    /// </summary>
    public class McpActionResult
    {
        /// <summary>
        /// Whether the action was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// The result data
        /// </summary>
        public JsonElement? Data { get; set; }

        /// <summary>
        /// The error message, if any
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// The ID of the action that this is a result for
        /// </summary>
        public string ActionId { get; set; } = string.Empty;

        /// <summary>
        /// The timestamp of the result
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Creates a successful result
        /// </summary>
        /// <param name="data">The result data</param>
        /// <param name="actionId">The ID of the action</param>
        /// <returns>A successful result</returns>
        public static McpActionResult CreateSuccess(JsonElement data, string actionId = "")
        {
            return new McpActionResult
            {
                Success = true,
                Data = data,
                ActionId = actionId
            };
        }

        /// <summary>
        /// Creates a failed result
        /// </summary>
        /// <param name="errorMessage">The error message</param>
        /// <param name="actionId">The ID of the action</param>
        /// <returns>A failed result</returns>
        public static McpActionResult CreateFailure(string errorMessage, string actionId = "")
        {
            return new McpActionResult
            {
                Success = false,
                ErrorMessage = errorMessage,
                ActionId = actionId
            };
        }
    }
}
