using System.Text.Json;

namespace TarsCli.Services.Agents;

/// <summary>
/// Interface for MCP agents
/// </summary>
public interface IAgent
{
    /// <summary>
    /// Gets the agent ID
    /// </summary>
    string Id { get; }

    /// <summary>
    /// Gets the agent name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the agent role
    /// </summary>
    string Role { get; }

    /// <summary>
    /// Gets the agent capabilities
    /// </summary>
    List<string> Capabilities { get; }

    /// <summary>
    /// Handles a request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    Task<JsonElement> HandleRequestAsync(JsonElement request);

    /// <summary>
    /// Initializes the agent
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    Task InitializeAsync();

    /// <summary>
    /// Shuts down the agent
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    Task ShutdownAsync();
}
