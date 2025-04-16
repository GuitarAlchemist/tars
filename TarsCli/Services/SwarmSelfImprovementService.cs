using System.Text.Json;
using Microsoft.Extensions.Configuration;
using TarsCli.Services.Agents;

namespace TarsCli.Services;

/// <summary>
/// Service for self-improvement using a swarm of MCP agents
/// </summary>
public class SwarmSelfImprovementService
{
    private readonly ILogger<SwarmSelfImprovementService> _logger;
    private readonly IConfiguration _configuration;
    private readonly TarsMcpSwarmService _swarmService;
    private readonly SelfImprovementService _selfImprovementService;
    private readonly DockerService _dockerService;
    private readonly AgentFactory _agentFactory;
    private readonly List<string> _agentIds = [];
    private readonly Dictionary<string, IAgent> _agents = new();
    private CancellationTokenSource _cancellationTokenSource;
    private Task _improvementTask;
    private bool _isRunning = false;

    /// <summary>
    /// Initializes a new instance of the SwarmSelfImprovementService class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="configuration">Configuration instance</param>
    /// <param name="swarmService">MCP swarm service instance</param>
    /// <param name="selfImprovementService">Self-improvement service instance</param>
    /// <param name="dockerService">Docker service instance</param>
    /// <param name="agentFactory">Agent factory instance</param>
    public SwarmSelfImprovementService(
        ILogger<SwarmSelfImprovementService> logger,
        IConfiguration configuration,
        TarsMcpSwarmService swarmService,
        SelfImprovementService selfImprovementService,
        DockerService dockerService,
        AgentFactory agentFactory)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _swarmService = swarmService ?? throw new ArgumentNullException(nameof(swarmService));
        _selfImprovementService = selfImprovementService ?? throw new ArgumentNullException(nameof(selfImprovementService));
        _dockerService = dockerService ?? throw new ArgumentNullException(nameof(dockerService));
        _agentFactory = agentFactory ?? throw new ArgumentNullException(nameof(agentFactory));
    }

    /// <summary>
    /// Starts the self-improvement process
    /// </summary>
    /// <param name="targetDirectories">Directories to target for improvement</param>
    /// <param name="agentCount">Number of agents to create</param>
    /// <param name="model">Model to use for improvement</param>
    /// <returns>True if the process was started successfully, false otherwise</returns>
    public async Task<bool> StartImprovementAsync(List<string> targetDirectories, int agentCount = 3, string model = "llama3")
    {
        try
        {
            if (_isRunning)
            {
                _logger.LogWarning("Self-improvement process is already running");
                return false;
            }

            // Check if Docker is running
            if (!await _dockerService.IsDockerRunning())
            {
                _logger.LogError("Docker is not running. Please start Docker first.");
                return false;
            }

            _logger.LogInformation($"Starting self-improvement process with {agentCount} agents");
            _logger.LogInformation($"Target directories: {string.Join(", ", targetDirectories)}");

            // Create the agents
            _agentIds.Clear();
            await CreateAgentsAsync(agentCount);

            // Start the improvement task
            _cancellationTokenSource = new CancellationTokenSource();
            _improvementTask = Task.Run(() => RunImprovementProcessAsync(targetDirectories, model, _cancellationTokenSource.Token));
            _isRunning = true;

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting self-improvement process");
            return false;
        }
    }

    /// <summary>
    /// Stops the self-improvement process
    /// </summary>
    /// <returns>True if the process was stopped successfully, false otherwise</returns>
    public async Task<bool> StopImprovementAsync()
    {
        try
        {
            if (!_isRunning)
            {
                _logger.LogWarning("Self-improvement process is not running");
                return false;
            }

            _logger.LogInformation("Stopping self-improvement process");

            // Cancel the improvement task
            _cancellationTokenSource?.Cancel();
            if (_improvementTask != null)
            {
                try
                {
                    await _improvementTask;
                }
                catch (OperationCanceledException)
                {
                    // Expected
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error waiting for improvement task to complete");
                }
            }

            // Stop all agents
            foreach (var agentId in _agentIds)
            {
                // Stop the local agent
                if (_agents.TryGetValue(agentId, out var agent))
                {
                    await agent.ShutdownAsync();
                }

                // Stop the MCP agent
                await _swarmService.StopAgentAsync(agentId);
            }

            // Clear the agent collections
            _agentIds.Clear();
            _agents.Clear();

            _isRunning = false;
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping self-improvement process");
            return false;
        }
    }

    /// <summary>
    /// Creates the agents for the self-improvement process
    /// </summary>
    /// <param name="agentCount">Number of agents to create</param>
    private async Task CreateAgentsAsync(int agentCount)
    {
        // Define agent roles
        var roles = new List<(string Name, string Role, List<string> Capabilities)>
        {
            ("CodeAnalyzer", "analyzer", ["analyze_code", "detect_issues", "suggest_improvements"]),
            ("CodeGenerator", "generator", ["generate_code", "refactor_code", "optimize_code"]),
            ("TestGenerator", "tester", ["generate_tests", "run_tests", "analyze_test_results"]),
            ("DocumentationGenerator", "documenter",
                ["generate_documentation", "update_documentation", "analyze_documentation"]),
            ("ProjectManager", "manager", ["manage_tasks", "prioritize_improvements", "track_progress"])
        };

        // Create the agents
        for (var i = 0; i < agentCount; i++)
        {
            var roleIndex = i % roles.Count;
            var (name, role, capabilities) = roles[roleIndex];
            var agentName = $"{name}-{i + 1}";
            var agentId = Guid.NewGuid().ToString();

            // Create the agent using the factory
            var agent = _agentFactory.CreateAgent(agentId, agentName, role, capabilities);
            await agent.InitializeAsync();

            // Add the agent to the collections
            _agentIds.Add(agentId);
            _agents[agentId] = agent;

            // Also create an MCP agent for remote communication
            var mcpAgent = await _swarmService.CreateAgentAsync(agentName, role, capabilities);
            _logger.LogInformation($"Created agent {agentName} (ID: {agentId}, Role: {role})");
        }
    }

    /// <summary>
    /// Runs the self-improvement process
    /// </summary>
    /// <param name="targetDirectories">Directories to target for improvement</param>
    /// <param name="model">Model to use for improvement</param>
    /// <param name="cancellationToken">Cancellation token</param>
    private async Task RunImprovementProcessAsync(List<string> targetDirectories, string model, CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation("Starting self-improvement process");

            // Create a directory for improvement artifacts
            var improvementDir = Path.Combine("data", "self-improvement", DateTime.Now.ToString("yyyyMMdd-HHmmss"));
            Directory.CreateDirectory(improvementDir);

            // Get all files in the target directories
            var files = GetFilesInDirectories(targetDirectories);
            _logger.LogInformation($"Found {files.Count} files to analyze");

            // Create a task queue
            var taskQueue = new Queue<string>(files);
            var completedTasks = new List<string>();
            var failedTasks = new List<string>();

            // Process files until the queue is empty or the task is cancelled
            while (taskQueue.Count > 0 && !cancellationToken.IsCancellationRequested)
            {
                // Get the next file
                var file = taskQueue.Dequeue();
                _logger.LogInformation($"Processing file: {file}");

                try
                {
                    // Analyze the file
                    var analysisResult = await AnalyzeFileAsync(file, model);

                    // Save the analysis result
                    var analysisPath = Path.Combine(improvementDir, $"{Path.GetFileNameWithoutExtension(file)}_analysis.json");
                    File.WriteAllText(analysisPath, JsonSerializer.Serialize(analysisResult, new JsonSerializerOptions { WriteIndented = true }));

                    // Check if improvements are needed
                    if (analysisResult.TryGetProperty("needs_improvement", out var needsImprovement) && needsImprovement.GetBoolean())
                    {
                        // Generate improvements
                        var improvementResult = await GenerateImprovementsAsync(file, analysisResult, model);

                        // Save the improvement result
                        var improvementPath = Path.Combine(improvementDir, $"{Path.GetFileNameWithoutExtension(file)}_improvements.json");
                        File.WriteAllText(improvementPath, JsonSerializer.Serialize(improvementResult, new JsonSerializerOptions { WriteIndented = true }));

                        // Apply improvements if auto-apply is enabled
                        var autoApply = _configuration.GetValue<bool>("Tars:SelfImprovement:AutoApply", false);
                        if (autoApply)
                        {
                            var applyResult = await ApplyImprovementsAsync(file, improvementResult, model);

                            // Save the apply result
                            var applyPath = Path.Combine(improvementDir, $"{Path.GetFileNameWithoutExtension(file)}_apply.json");
                            File.WriteAllText(applyPath, JsonSerializer.Serialize(applyResult, new JsonSerializerOptions { WriteIndented = true }));
                        }
                    }

                    completedTasks.Add(file);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error processing file: {file}");
                    failedTasks.Add(file);
                }

                // Wait a bit to avoid overwhelming the system
                await Task.Delay(1000, cancellationToken);
            }

            _logger.LogInformation($"Self-improvement process completed. Processed {completedTasks.Count} files successfully, {failedTasks.Count} files failed.");
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("Self-improvement process was cancelled");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running self-improvement process");
        }
        finally
        {
            _isRunning = false;
        }
    }

    /// <summary>
    /// Gets all files in the specified directories
    /// </summary>
    /// <param name="directories">Directories to search</param>
    /// <returns>List of file paths</returns>
    private List<string> GetFilesInDirectories(List<string> directories)
    {
        var files = new List<string>();
        var extensions = new[] { ".cs", ".fs", ".fsx", ".csproj", ".fsproj", ".md", ".json", ".yml", ".yaml" };

        foreach (var directory in directories)
        {
            if (!Directory.Exists(directory))
            {
                _logger.LogWarning($"Directory not found: {directory}");
                continue;
            }

            foreach (var extension in extensions)
            {
                files.AddRange(Directory.GetFiles(directory, $"*{extension}", SearchOption.AllDirectories));
            }
        }

        return files;
    }

    /// <summary>
    /// Analyzes a file using the code analyzer agent
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="model">Model to use for analysis</param>
    /// <returns>Analysis result as a JsonElement</returns>
    private async Task<JsonElement> AnalyzeFileAsync(string filePath, string model)
    {
        // Find a code analyzer agent
        var analyzerAgent = _agents.Values.FirstOrDefault(a => a.Role == "analyzer");

        if (analyzerAgent == null)
        {
            throw new InvalidOperationException("No code analyzer agent found");
        }

        // Read the file content
        var fileContent = await File.ReadAllTextAsync(filePath);
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');

        // Create the request
        var requestObj = new
        {
            action = "code",
            operation = "analyze",
            file_path = filePath,
            file_content = fileContent,
            file_type = fileExtension,
            model = model
        };

        var requestJson = JsonSerializer.Serialize(requestObj);
        var request = JsonSerializer.Deserialize<JsonElement>(requestJson);

        // Send the request to the agent
        return await analyzerAgent.HandleRequestAsync(request);
    }

    /// <summary>
    /// Generates improvements for a file using the code generator agent
    /// </summary>
    /// <param name="filePath">Path to the file to improve</param>
    /// <param name="analysisResult">Analysis result from the code analyzer</param>
    /// <param name="model">Model to use for improvement</param>
    /// <returns>Improvement result as a JsonElement</returns>
    private async Task<JsonElement> GenerateImprovementsAsync(string filePath, JsonElement analysisResult, string model)
    {
        // Find a code generator agent
        var generatorAgent = _agents.Values.FirstOrDefault(a => a.Role == "generator");

        if (generatorAgent == null)
        {
            throw new InvalidOperationException("No code generator agent found");
        }

        // Read the file content
        var fileContent = await File.ReadAllTextAsync(filePath);
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');

        // Create the request
        var requestObj = new
        {
            action = "code",
            operation = "improve",
            file_path = filePath,
            file_content = fileContent,
            file_type = fileExtension,
            analysis = analysisResult.ToString(),
            model = model
        };

        var requestJson = JsonSerializer.Serialize(requestObj);
        var request = JsonSerializer.Deserialize<JsonElement>(requestJson);

        // Send the request to the agent
        return await generatorAgent.HandleRequestAsync(request);
    }

    /// <summary>
    /// Applies improvements to a file
    /// </summary>
    /// <param name="filePath">Path to the file to improve</param>
    /// <param name="improvementResult">Improvement result from the code generator</param>
    /// <param name="model">Model to use for improvement</param>
    /// <returns>Apply result as a JsonElement</returns>
    private async Task<JsonElement> ApplyImprovementsAsync(string filePath, JsonElement improvementResult, string model)
    {
        // Check if the improvement result contains improved content
        if (!improvementResult.TryGetProperty("improved_content", out var improvedContent))
        {
            throw new InvalidOperationException("Improvement result does not contain improved content");
        }

        // Create a backup of the original file
        var backupPath = $"{filePath}.bak";
        File.Copy(filePath, backupPath, true);

        // Write the improved content to the file
        await File.WriteAllTextAsync(filePath, improvedContent.GetString());

        // Return a result
        var resultObj = new
        {
            success = true,
            file_path = filePath,
            backup_path = backupPath,
            timestamp = DateTime.UtcNow
        };

        var resultJson = JsonSerializer.Serialize(resultObj);
        return JsonSerializer.Deserialize<JsonElement>(resultJson);
    }

    /// <summary>
    /// Gets the status of the self-improvement process
    /// </summary>
    /// <returns>Status information as a JsonElement</returns>
    public JsonElement GetStatus()
    {
        var mcpAgents = _swarmService.GetAllAgents();
        var runningMcpAgents = mcpAgents.Count(a => a.Status == "running");

        var localAgents = _agents.Values.ToList();

        var statusObj = new
        {
            is_running = _isRunning,
            agent_count = _agentIds.Count,
            running_mcp_agents = runningMcpAgents,
            local_agents = localAgents.Select(a => new
            {
                id = a.Id,
                name = a.Name,
                role = a.Role,
                capabilities = a.Capabilities
            }).ToList(),
            mcp_agents = _agentIds.Select(id => _swarmService.GetAgent(id)).ToList()
        };

        var statusJson = JsonSerializer.Serialize(statusObj);
        return JsonSerializer.Deserialize<JsonElement>(statusJson);
    }
}