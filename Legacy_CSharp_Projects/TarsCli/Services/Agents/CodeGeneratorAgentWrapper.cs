using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Agents;

/// <summary>
/// Wrapper for the code generator agent
/// </summary>
public class CodeGeneratorAgentWrapper : AgentBase
{
    private readonly CodeGeneratorAgent _codeGeneratorAgent;

    /// <summary>
    /// Initializes a new instance of the CodeGeneratorAgentWrapper class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="role">Agent role</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <param name="codeGeneratorAgent">Code generator agent</param>
    public CodeGeneratorAgentWrapper(
        ILogger<CodeGeneratorAgentWrapper> logger,
        string id,
        string name,
        string role,
        List<string> capabilities,
        CodeGeneratorAgent codeGeneratorAgent)
        : base(logger, id, name, role, capabilities)
    {
        _codeGeneratorAgent = codeGeneratorAgent ?? throw new ArgumentNullException(nameof(codeGeneratorAgent));
    }

    /// <summary>
    /// Handles a request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public override Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        return _codeGeneratorAgent.HandleRequestAsync(request);
    }
}
