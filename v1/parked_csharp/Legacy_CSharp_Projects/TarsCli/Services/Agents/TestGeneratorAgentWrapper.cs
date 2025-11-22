using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Agents;

/// <summary>
/// Wrapper for the test generator agent
/// </summary>
public class TestGeneratorAgentWrapper : AgentBase
{
    private readonly TestGeneratorAgent _testGeneratorAgent;

    /// <summary>
    /// Initializes a new instance of the TestGeneratorAgentWrapper class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="role">Agent role</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <param name="testGeneratorAgent">Test generator agent</param>
    public TestGeneratorAgentWrapper(
        ILogger<TestGeneratorAgentWrapper> logger,
        string id,
        string name,
        string role,
        List<string> capabilities,
        TestGeneratorAgent testGeneratorAgent)
        : base(logger, id, name, role, capabilities)
    {
        _testGeneratorAgent = testGeneratorAgent ?? throw new ArgumentNullException(nameof(testGeneratorAgent));
    }

    /// <summary>
    /// Handles a request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public override Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        return _testGeneratorAgent.HandleRequestAsync(request);
    }
}
