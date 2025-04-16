using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Agents;

/// <summary>
/// Factory for creating MCP agents
/// </summary>
public class AgentFactory
{
    private readonly ILogger<AgentFactory> _logger;
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the AgentFactory class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="serviceProvider">Service provider</param>
    public AgentFactory(ILogger<AgentFactory> logger, IServiceProvider serviceProvider)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
    }

    /// <summary>
    /// Creates an agent
    /// </summary>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="role">Agent role</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <returns>The created agent</returns>
    public IAgent CreateAgent(string id, string name, string role, List<string> capabilities = null)
    {
        _logger.LogInformation($"Creating agent {name} (ID: {id}, Role: {role})");

        // Create the agent based on the role
        return role.ToLower() switch
        {
            "analyzer" => CreateCodeAnalyzerAgent(id, name, capabilities),
            "generator" => CreateCodeGeneratorAgent(id, name, capabilities),
            "tester" => CreateTestGeneratorAgent(id, name, capabilities),
            "documenter" => CreateDocumentationGeneratorAgent(id, name, capabilities),
            "manager" => CreateProjectManagerAgent(id, name, capabilities),
            _ => throw new ArgumentException($"Unknown agent role: {role}")
        };
    }

    /// <summary>
    /// Creates a code analyzer agent
    /// </summary>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <returns>The created agent</returns>
    private IAgent CreateCodeAnalyzerAgent(string id, string name, List<string> capabilities)
    {
        var logger = _serviceProvider.GetRequiredService<ILogger<CodeAnalyzerAgentWrapper>>();
        var codeAnalyzerAgent = _serviceProvider.GetRequiredService<CodeAnalyzerAgent>();
        return new CodeAnalyzerAgentWrapper(logger, id, name, "analyzer", capabilities, codeAnalyzerAgent);
    }

    /// <summary>
    /// Creates a code generator agent
    /// </summary>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <returns>The created agent</returns>
    private IAgent CreateCodeGeneratorAgent(string id, string name, List<string> capabilities)
    {
        var logger = _serviceProvider.GetRequiredService<ILogger<CodeGeneratorAgentWrapper>>();
        var codeGeneratorAgent = _serviceProvider.GetRequiredService<CodeGeneratorAgent>();
        return new CodeGeneratorAgentWrapper(logger, id, name, "generator", capabilities, codeGeneratorAgent);
    }

    /// <summary>
    /// Creates a test generator agent
    /// </summary>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <returns>The created agent</returns>
    private IAgent CreateTestGeneratorAgent(string id, string name, List<string> capabilities)
    {
        var logger = _serviceProvider.GetRequiredService<ILogger<TestGeneratorAgentWrapper>>();
        var testGeneratorAgent = _serviceProvider.GetRequiredService<TestGeneratorAgent>();
        return new TestGeneratorAgentWrapper(logger, id, name, "tester", capabilities, testGeneratorAgent);
    }

    /// <summary>
    /// Creates a documentation generator agent
    /// </summary>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <returns>The created agent</returns>
    private IAgent CreateDocumentationGeneratorAgent(string id, string name, List<string> capabilities)
    {
        var logger = _serviceProvider.GetRequiredService<ILogger<DocumentationGeneratorAgentWrapper>>();
        var documentationGeneratorAgent = _serviceProvider.GetRequiredService<DocumentationGeneratorAgent>();
        return new DocumentationGeneratorAgentWrapper(logger, id, name, "documenter", capabilities, documentationGeneratorAgent);
    }

    /// <summary>
    /// Creates a project manager agent
    /// </summary>
    /// <param name="id">Agent ID</param>
    /// <param name="name">Agent name</param>
    /// <param name="capabilities">Agent capabilities</param>
    /// <returns>The created agent</returns>
    private IAgent CreateProjectManagerAgent(string id, string name, List<string> capabilities)
    {
        var logger = _serviceProvider.GetRequiredService<ILogger<ProjectManagerAgentWrapper>>();
        var projectManagerAgent = _serviceProvider.GetRequiredService<ProjectManagerAgent>();
        return new ProjectManagerAgentWrapper(logger, id, name, "manager", capabilities, projectManagerAgent);
    }
}
