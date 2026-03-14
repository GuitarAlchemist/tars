using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Agents;

/// <summary>
/// Wrapper for the project manager agent
/// </summary>
public class ProjectManagerAgentWrapper : AgentBase
{
    private readonly ProjectManagerAgent _projectManagerAgent;

    /// <summary>
    /// Initializes a new instance of the ProjectManagerAgentWrapper class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="role">Agent role</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <param name="projectManagerAgent">Project manager agent</param>
    public ProjectManagerAgentWrapper(
        ILogger<ProjectManagerAgentWrapper> logger,
        string id,
        string name,
        string role,
        List<string> capabilities,
        ProjectManagerAgent projectManagerAgent)
        : base(logger, id, name, role, capabilities)
    {
        _projectManagerAgent = projectManagerAgent ?? throw new ArgumentNullException(nameof(projectManagerAgent));
    }

    /// <summary>
    /// Handles a request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public override Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        return _projectManagerAgent.HandleRequestAsync(request);
    }
}
