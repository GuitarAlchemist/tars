using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Agents;

/// <summary>
/// Base class for MCP agents
/// </summary>
public abstract class AgentBase : IAgent
{
    private readonly ILogger _logger;

    /// <summary>
    /// Gets the agent ID
    /// </summary>
    public string Id { get; }

    /// <summary>
    /// Gets the agent name
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the agent role
    /// </summary>
    public string Role { get; }

    /// <summary>
    /// Gets the agent capabilities
    /// </summary>
    public List<string> Capabilities { get; }

    /// <summary>
    /// Initializes a new instance of the AgentBase class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="role">Agent role</param>
    /// <param name="capabilities">Agent capabilities</param>
    protected AgentBase(ILogger logger, string id, string name, string role, List<string> capabilities)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        Id = id ?? throw new ArgumentNullException(nameof(id));
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Role = role ?? throw new ArgumentNullException(nameof(role));
        Capabilities = capabilities ?? new List<string>();
    }

    /// <summary>
    /// Handles a request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public abstract Task<JsonElement> HandleRequestAsync(JsonElement request);

    /// <summary>
    /// Initializes the agent
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public virtual async Task InitializeAsync()
    {
        _logger.LogInformation($"Initializing agent {Name} (ID: {Id}, Role: {Role})");
        await Task.CompletedTask;
    }

    /// <summary>
    /// Shuts down the agent
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public virtual async Task ShutdownAsync()
    {
        _logger.LogInformation($"Shutting down agent {Name} (ID: {Id}, Role: {Role})");
        await Task.CompletedTask;
    }

    /// <summary>
    /// Creates an error response
    /// </summary>
    /// <param name="message">The error message</param>
    /// <returns>The error response</returns>
    protected JsonElement CreateErrorResponse(string message)
    {
        var responseObj = new
        {
            success = false,
            error = message
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }
}
