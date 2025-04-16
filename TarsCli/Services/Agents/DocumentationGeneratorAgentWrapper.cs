using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Agents;

/// <summary>
/// Wrapper for the documentation generator agent
/// </summary>
public class DocumentationGeneratorAgentWrapper : AgentBase
{
    private readonly DocumentationGeneratorAgent _documentationGeneratorAgent;

    /// <summary>
    /// Initializes a new instance of the DocumentationGeneratorAgentWrapper class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="role">Agent role</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <param name="documentationGeneratorAgent">Documentation generator agent</param>
    public DocumentationGeneratorAgentWrapper(
        ILogger<DocumentationGeneratorAgentWrapper> logger,
        string id,
        string name,
        string role,
        List<string> capabilities,
        DocumentationGeneratorAgent documentationGeneratorAgent)
        : base(logger, id, name, role, capabilities)
    {
        _documentationGeneratorAgent = documentationGeneratorAgent ?? throw new ArgumentNullException(nameof(documentationGeneratorAgent));
    }

    /// <summary>
    /// Handles a request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public override Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        return _documentationGeneratorAgent.HandleRequestAsync(request);
    }
}
