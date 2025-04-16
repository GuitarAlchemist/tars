using Microsoft.Extensions.Configuration;
using TarsEngine.A2A;
using Task = System.Threading.Tasks.Task;

namespace TarsCli.Services;

/// <summary>
/// Service for TARS A2A protocol implementation
/// </summary>
public class TarsA2AService
{
    private readonly ILogger<TarsA2AService> _logger;
    private readonly IConfiguration _configuration;
    private readonly A2AServer _a2aServer;
    private readonly A2AMcpBridge _a2aMcpBridge;
    private readonly TarsMcpService _tarsMcpService;

    /// <summary>
    /// Initializes a new instance of the TarsA2AService class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="configuration">Configuration instance</param>
    /// <param name="tarsMcpService">TARS MCP service instance</param>
    public TarsA2AService(
        ILogger<TarsA2AService> logger,
        IConfiguration configuration,
        TarsMcpService tarsMcpService)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _tarsMcpService = tarsMcpService ?? throw new ArgumentNullException(nameof(tarsMcpService));

        // Create the agent card
        var agentCard = CreateAgentCard();

        // Create the A2A server
        var a2aServerLogger = new LoggerAdapter<A2AServer>(logger);
        _a2aServer = new A2AServer(agentCard, a2aServerLogger);

        // Create the A2A-MCP bridge
        var a2aMcpBridgeLogger = new LoggerAdapter<A2AMcpBridge>(logger);
        _a2aMcpBridge = new A2AMcpBridge(
            a2aMcpBridgeLogger,
            _tarsMcpService.McpService,
            _a2aServer);
    }

    /// <summary>
    /// Starts the TARS A2A service
    /// </summary>
    public Task StartAsync()
    {
        try
        {
            // Initialize the A2A-MCP bridge
            _a2aMcpBridge.Initialize();

            // Start the A2A server
            _a2aServer.Start();

            _logger.LogInformation("TARS A2A service started");
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting TARS A2A service");
            throw;
        }
    }

    /// <summary>
    /// Stops the TARS A2A service
    /// </summary>
    public Task StopAsync()
    {
        try
        {
            // Stop the A2A server
            _a2aServer.Stop();

            _logger.LogInformation("TARS A2A service stopped");
            return Task.CompletedTask;
        }
        catch (System.Net.HttpListenerException ex) when (ex.ErrorCode == 995)
        {
            // This is expected during shutdown, no need to log an error
            _logger.LogInformation("TARS A2A service stopped");
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping TARS A2A service");
            throw;
        }
    }

    /// <summary>
    /// Creates the TARS agent card
    /// </summary>
    private AgentCard CreateAgentCard()
    {
        // Get configuration values
        var port = _configuration.GetValue<int>("Tars:A2A:Port", 8998);
        var host = _configuration.GetValue<string>("Tars:A2A:Host", "localhost");
        var url = $"http://{host}:{port}/";

        // Create the agent card
        var agentCard = new AgentCard
        {
            Name = "TARS Agent",
            Description = "TARS Agent with A2A protocol support",
            Url = url,
            Version = typeof(TarsA2AService).Assembly.GetName().Version.ToString(),
            DocumentationUrl = "https://github.com/GuitarAlchemist/tars",
            Provider = new AgentProvider
            {
                Organization = "TARS",
                Url = "https://github.com/GuitarAlchemist/tars"
            },
            Capabilities = new AgentCapabilities
            {
                Streaming = true,
                PushNotifications = true,
                StateTransitionHistory = true
            },
            Authentication = null, // No authentication required for now
            DefaultInputModes = ["text"],
            DefaultOutputModes = ["text"],
            Skills =
            [
                new()
                {
                    Id = "code_generation",
                    Name = "Code Generation",
                    Description = "Generate code based on natural language descriptions",
                    Tags = ["code", "generation", "programming"],
                    Examples =
                    [
                        "Generate a C# class for a customer entity",
                        "Create a function to calculate Fibonacci numbers"
                    ]
                },

                new()
                {
                    Id = "code_analysis",
                    Name = "Code Analysis",
                    Description = "Analyze code for quality, complexity, and issues",
                    Tags = ["code", "analysis", "quality"],
                    Examples =
                    [
                        "Analyze this C# code for potential issues",
                        "Review this function for performance problems"
                    ]
                },

                new()
                {
                    Id = "metascript_execution",
                    Name = "Metascript Execution",
                    Description = "Execute TARS metascripts",
                    Tags = ["metascript", "execution", "automation"],
                    Examples =
                    [
                        "Execute this metascript to analyze a project",
                        "Run a metascript to generate documentation"
                    ]
                },

                new()
                {
                    Id = "knowledge_extraction",
                    Name = "Knowledge Extraction",
                    Description = "Extract knowledge from documents and code",
                    Tags = ["knowledge", "extraction", "documentation"],
                    Examples =
                    [
                        "Extract key concepts from this documentation",
                        "Identify important patterns in this codebase"
                    ]
                },

                new()
                {
                    Id = "self_improvement",
                    Name = "Self Improvement",
                    Description = "Improve TARS capabilities through self-analysis",
                    Tags = ["self-improvement", "learning", "optimization"],
                    Examples =
                    [
                        "Analyze TARS performance and suggest improvements",
                        "Identify areas where TARS can be enhanced"
                    ]
                }
            ]
        };

        return agentCard;
    }
}