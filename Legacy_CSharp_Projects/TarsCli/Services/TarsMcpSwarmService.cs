using System.Text.Json;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

/// <summary>
/// Service for managing a swarm of TARS MCP servers in Docker containers
/// </summary>
public class TarsMcpSwarmService
{
    private readonly ILogger<TarsMcpSwarmService> _logger;
    private readonly IConfiguration _configuration;
    private readonly DockerService _dockerService;
    private readonly McpService _mcpService;
    private readonly Dictionary<string, AgentInfo> _agents = new();
    private readonly string _swarmConfigPath;
    private readonly string _dockerComposeTemplatePath;
    private readonly string _dockerComposeOutputDir;
    private int _nextAgentPort = 9001; // Start from 9001 since 9000 is the main MCP server

    /// <summary>
    /// Represents information about a TARS agent in the swarm
    /// </summary>
    public class AgentInfo
    {
        /// <summary>
        /// Unique identifier for the agent
        /// </summary>
        public string Id { get; set; }

        /// <summary>
        /// Name of the agent
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Role of the agent (e.g., "code_analyzer", "test_generator", etc.)
        /// </summary>
        public string Role { get; set; }

        /// <summary>
        /// Port the agent is listening on
        /// </summary>
        public int Port { get; set; }

        /// <summary>
        /// Docker container name
        /// </summary>
        public string ContainerName { get; set; }

        /// <summary>
        /// Docker compose file path
        /// </summary>
        public string DockerComposePath { get; set; }

        /// <summary>
        /// Status of the agent (e.g., "running", "stopped", etc.)
        /// </summary>
        public string Status { get; set; }

        /// <summary>
        /// Time the agent was created
        /// </summary>
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// Time the agent was last active
        /// </summary>
        public DateTime LastActiveAt { get; set; }

        /// <summary>
        /// Capabilities of the agent
        /// </summary>
        public List<string> Capabilities { get; set; }

        /// <summary>
        /// Additional metadata for the agent
        /// </summary>
        public Dictionary<string, string> Metadata { get; set; }
    }

    /// <summary>
    /// Initializes a new instance of the TarsMcpSwarmService class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="configuration">Configuration instance</param>
    /// <param name="dockerService">Docker service instance</param>
    /// <param name="mcpService">MCP service instance</param>
    public TarsMcpSwarmService(
        ILogger<TarsMcpSwarmService> logger,
        IConfiguration configuration,
        DockerService dockerService,
        McpService mcpService)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _dockerService = dockerService ?? throw new ArgumentNullException(nameof(dockerService));
        _mcpService = mcpService ?? throw new ArgumentNullException(nameof(mcpService));

        // Get configuration values
        _swarmConfigPath = _configuration["Tars:McpSwarm:ConfigPath"] ?? "config/mcp-swarm.json";
        _dockerComposeTemplatePath = _configuration["Tars:McpSwarm:DockerComposeTemplatePath"] ?? "templates/docker-compose-mcp-agent.yml";
        _dockerComposeOutputDir = _configuration["Tars:McpSwarm:DockerComposeOutputDir"] ?? "docker/mcp-agents";

        // Create the output directory if it doesn't exist
        Directory.CreateDirectory(_dockerComposeOutputDir);

        // Load existing agents
        LoadAgents();
    }

    /// <summary>
    /// Loads existing agents from the configuration file
    /// </summary>
    private void LoadAgents()
    {
        try
        {
            if (File.Exists(_swarmConfigPath))
            {
                var json = File.ReadAllText(_swarmConfigPath);
                var agents = JsonSerializer.Deserialize<List<AgentInfo>>(json);
                if (agents != null)
                {
                    foreach (var agent in agents)
                    {
                        _agents[agent.Id] = agent;
                        if (agent.Port >= _nextAgentPort)
                        {
                            _nextAgentPort = agent.Port + 1;
                        }
                    }
                    _logger.LogInformation($"Loaded {_agents.Count} agents from configuration");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading agents from configuration");
        }
    }

    /// <summary>
    /// Saves the current agents to the configuration file
    /// </summary>
    private void SaveAgents()
    {
        try
        {
            var json = JsonSerializer.Serialize(_agents.Values.ToList(), new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(Path.GetDirectoryName(_swarmConfigPath));
            File.WriteAllText(_swarmConfigPath, json);
            _logger.LogInformation($"Saved {_agents.Count} agents to configuration");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving agents to configuration");
        }
    }

    /// <summary>
    /// Creates a new agent in the swarm
    /// </summary>
    /// <param name="name">Name of the agent</param>
    /// <param name="role">Role of the agent</param>
    /// <param name="capabilities">Capabilities of the agent</param>
    /// <param name="metadata">Additional metadata for the agent</param>
    /// <returns>The created agent info</returns>
    public async Task<AgentInfo> CreateAgentAsync(string name, string role, List<string> capabilities = null, Dictionary<string, string> metadata = null)
    {
        try
        {
            // Check if Docker is running
            if (!await _dockerService.IsDockerRunning())
            {
                throw new InvalidOperationException("Docker is not running. Please start Docker first.");
            }

            // Create a unique ID for the agent
            var id = Guid.NewGuid().ToString("N");
            var port = _nextAgentPort++;
            var containerName = $"tars-mcp-agent-{id.Substring(0, 8)}";
            var dockerComposePath = Path.Combine(_dockerComposeOutputDir, $"docker-compose-{id.Substring(0, 8)}.yml");

            // Create the agent info
            var agent = new AgentInfo
            {
                Id = id,
                Name = name,
                Role = role,
                Port = port,
                ContainerName = containerName,
                DockerComposePath = dockerComposePath,
                Status = "creating",
                CreatedAt = DateTime.UtcNow,
                LastActiveAt = DateTime.UtcNow,
                Capabilities = capabilities ?? new List<string>(),
                Metadata = metadata ?? new Dictionary<string, string>()
            };

            // Add the agent to the dictionary
            _agents[id] = agent;
            SaveAgents();

            // Create the Docker Compose file
            await CreateDockerComposeFileAsync(agent);

            // Start the agent container
            if (await _dockerService.StartContainer(dockerComposePath, containerName))
            {
                agent.Status = "running";
                SaveAgents();
                _logger.LogInformation($"Created and started agent {name} (ID: {id}) on port {port}");
            }
            else
            {
                agent.Status = "failed";
                SaveAgents();
                _logger.LogError($"Failed to start agent {name} (ID: {id})");
            }

            return agent;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error creating agent {name}");
            throw;
        }
    }

    /// <summary>
    /// Creates a Docker Compose file for an agent
    /// </summary>
    /// <param name="agent">The agent info</param>
    private async Task CreateDockerComposeFileAsync(AgentInfo agent)
    {
        try
        {
            // Read the template
            var template = await File.ReadAllTextAsync(_dockerComposeTemplatePath);

            // Replace placeholders
            var content = template
                .Replace("{{AGENT_ID}}", agent.Id)
                .Replace("{{AGENT_NAME}}", agent.Name)
                .Replace("{{AGENT_ROLE}}", agent.Role)
                .Replace("{{AGENT_PORT}}", agent.Port.ToString())
                .Replace("{{CONTAINER_NAME}}", agent.ContainerName);

            // Write the Docker Compose file
            Directory.CreateDirectory(Path.GetDirectoryName(agent.DockerComposePath));
            await File.WriteAllTextAsync(agent.DockerComposePath, content);
            _logger.LogInformation($"Created Docker Compose file for agent {agent.Name} at {agent.DockerComposePath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error creating Docker Compose file for agent {agent.Name}");
            throw;
        }
    }

    /// <summary>
    /// Starts an agent in the swarm
    /// </summary>
    /// <param name="agentId">ID of the agent to start</param>
    /// <returns>True if the agent was started successfully, false otherwise</returns>
    public async Task<bool> StartAgentAsync(string agentId)
    {
        try
        {
            if (!_agents.TryGetValue(agentId, out var agent))
            {
                _logger.LogError($"Agent with ID {agentId} not found");
                return false;
            }

            // Check if Docker is running
            if (!await _dockerService.IsDockerRunning())
            {
                _logger.LogError("Docker is not running. Please start Docker first.");
                return false;
            }

            // Start the agent container
            if (await _dockerService.StartContainer(agent.DockerComposePath, agent.ContainerName))
            {
                agent.Status = "running";
                agent.LastActiveAt = DateTime.UtcNow;
                SaveAgents();
                _logger.LogInformation($"Started agent {agent.Name} (ID: {agentId})");
                return true;
            }
            else
            {
                _logger.LogError($"Failed to start agent {agent.Name} (ID: {agentId})");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error starting agent with ID {agentId}");
            return false;
        }
    }

    /// <summary>
    /// Stops an agent in the swarm
    /// </summary>
    /// <param name="agentId">ID of the agent to stop</param>
    /// <returns>True if the agent was stopped successfully, false otherwise</returns>
    public async Task<bool> StopAgentAsync(string agentId)
    {
        try
        {
            if (!_agents.TryGetValue(agentId, out var agent))
            {
                _logger.LogError($"Agent with ID {agentId} not found");
                return false;
            }

            // Check if Docker is running
            if (!await _dockerService.IsDockerRunning())
            {
                _logger.LogError("Docker is not running. Please start Docker first.");
                return false;
            }

            // Stop the agent container
            if (await _dockerService.StopContainer(agent.DockerComposePath, agent.ContainerName))
            {
                agent.Status = "stopped";
                SaveAgents();
                _logger.LogInformation($"Stopped agent {agent.Name} (ID: {agentId})");
                return true;
            }
            else
            {
                _logger.LogError($"Failed to stop agent {agent.Name} (ID: {agentId})");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error stopping agent with ID {agentId}");
            return false;
        }
    }

    /// <summary>
    /// Removes an agent from the swarm
    /// </summary>
    /// <param name="agentId">ID of the agent to remove</param>
    /// <returns>True if the agent was removed successfully, false otherwise</returns>
    public async Task<bool> RemoveAgentAsync(string agentId)
    {
        try
        {
            if (!_agents.TryGetValue(agentId, out var agent))
            {
                _logger.LogError($"Agent with ID {agentId} not found");
                return false;
            }

            // Check if Docker is running
            if (!await _dockerService.IsDockerRunning())
            {
                _logger.LogError("Docker is not running. Please start Docker first.");
                return false;
            }

            // Stop and remove the agent container
            await _dockerService.StopContainer(agent.DockerComposePath, agent.ContainerName);

            // Remove the Docker Compose file
            if (File.Exists(agent.DockerComposePath))
            {
                File.Delete(agent.DockerComposePath);
            }

            // Remove the agent from the dictionary
            _agents.Remove(agentId);
            SaveAgents();
            _logger.LogInformation($"Removed agent {agent.Name} (ID: {agentId})");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error removing agent with ID {agentId}");
            return false;
        }
    }

    /// <summary>
    /// Gets all agents in the swarm
    /// </summary>
    /// <returns>List of all agents</returns>
    public List<AgentInfo> GetAllAgents()
    {
        return _agents.Values.ToList();
    }

    /// <summary>
    /// Gets an agent by ID
    /// </summary>
    /// <param name="agentId">ID of the agent to get</param>
    /// <returns>The agent info, or null if not found</returns>
    public AgentInfo GetAgent(string agentId)
    {
        return _agents.TryGetValue(agentId, out var agent) ? agent : null;
    }

    /// <summary>
    /// Updates the status of all agents in the swarm
    /// </summary>
    public async Task UpdateAgentStatusesAsync()
    {
        try
        {
            // Check if Docker is running
            if (!await _dockerService.IsDockerRunning())
            {
                _logger.LogWarning("Docker is not running. Cannot update agent statuses.");
                return;
            }

            foreach (var agent in _agents.Values)
            {
                try
                {
                    var status = await _dockerService.GetContainerStatus(agent.ContainerName);
                    agent.Status = status;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error updating status for agent {agent.Name} (ID: {agent.Id})");
                }
            }

            SaveAgents();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating agent statuses");
        }
    }

    /// <summary>
    /// Sends a request to an agent
    /// </summary>
    /// <param name="agentId">ID of the agent to send the request to</param>
    /// <param name="request">The request to send</param>
    /// <returns>The response from the agent</returns>
    public async Task<JsonElement> SendRequestToAgentAsync(string agentId, JsonElement request)
    {
        try
        {
            if (!_agents.TryGetValue(agentId, out var agent))
            {
                throw new ArgumentException($"Agent with ID {agentId} not found");
            }

            if (agent.Status != "running")
            {
                throw new InvalidOperationException($"Agent {agent.Name} (ID: {agentId}) is not running");
            }

            // Update the last active time
            agent.LastActiveAt = DateTime.UtcNow;
            SaveAgents();

            // Extract action from the request
            var action = "code";
            if (request.TryGetProperty("action", out var actionElement))
            {
                action = actionElement.GetString();
            }

            // Send the request to the agent's MCP server
            var url = $"http://localhost:{agent.Port}/";
            return await _mcpService.SendRequestAsync(url, action, request);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error sending request to agent with ID {agentId}");
            throw;
        }
    }

    /// <summary>
    /// Starts all agents in the swarm
    /// </summary>
    /// <returns>True if all agents were started successfully, false otherwise</returns>
    public async Task<bool> StartAllAgentsAsync()
    {
        var success = true;
        foreach (var agentId in _agents.Keys)
        {
            if (!await StartAgentAsync(agentId))
            {
                success = false;
            }
        }
        return success;
    }

    /// <summary>
    /// Stops all agents in the swarm
    /// </summary>
    /// <returns>True if all agents were stopped successfully, false otherwise</returns>
    public async Task<bool> StopAllAgentsAsync()
    {
        var success = true;
        foreach (var agentId in _agents.Keys)
        {
            if (!await StopAgentAsync(agentId))
            {
                success = false;
            }
        }
        return success;
    }
}