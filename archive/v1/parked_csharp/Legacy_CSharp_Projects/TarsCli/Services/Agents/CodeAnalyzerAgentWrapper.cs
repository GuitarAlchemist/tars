using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Agents;

/// <summary>
/// Wrapper for the code analyzer agent
/// </summary>
public class CodeAnalyzerAgentWrapper : AgentBase
{
    private readonly CodeAnalyzerAgent _codeAnalyzerAgent;

    /// <summary>
    /// Initializes a new instance of the CodeAnalyzerAgentWrapper class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="role">Agent role</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <param name="codeAnalyzerAgent">Code analyzer agent</param>
    public CodeAnalyzerAgentWrapper(
        ILogger<CodeAnalyzerAgentWrapper> logger,
        string id,
        string name,
        string role,
        List<string> capabilities,
        CodeAnalyzerAgent codeAnalyzerAgent)
        : base(logger, id, name, role, capabilities)
    {
        _codeAnalyzerAgent = codeAnalyzerAgent ?? throw new ArgumentNullException(nameof(codeAnalyzerAgent));
    }

    /// <summary>
    /// Handles a request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public override Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        return _codeAnalyzerAgent.HandleRequestAsync(request);
    }
}
