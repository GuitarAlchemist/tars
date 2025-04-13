using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for managing TARS replicas in Docker containers
    /// </summary>
    public class TarsReplicaManager
    {
        private readonly ILogger<TarsReplicaManager> _logger;
        private readonly IConfiguration _configuration;
        private readonly DockerService _dockerService;
        private readonly McpService _mcpService;
        private readonly Dictionary<string, ReplicaInfo> _replicas = new Dictionary<string, ReplicaInfo>();
        private readonly string _replicaConfigPath;
        private readonly string _dockerComposeTemplatePath;
        private readonly string _dockerComposeOutputDir;
        private int _nextReplicaPort = 9001; // Start from 9001 since 9000 is the main MCP server

        /// <summary>
        /// Represents information about a TARS replica
        /// </summary>
        public class ReplicaInfo
        {
            /// <summary>
            /// Unique identifier for the replica
            /// </summary>
            public string Id { get; set; }

            /// <summary>
            /// Name of the replica
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// Role of the replica (e.g., "analyzer", "generator", "tester", "coordinator")
            /// </summary>
            public string Role { get; set; }

            /// <summary>
            /// Port the replica is listening on
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
            /// Status of the replica (e.g., "running", "stopped", etc.)
            /// </summary>
            public string Status { get; set; }

            /// <summary>
            /// Time the replica was created
            /// </summary>
            public DateTime CreatedAt { get; set; }

            /// <summary>
            /// Time the replica was last active
            /// </summary>
            public DateTime LastActiveAt { get; set; }

            /// <summary>
            /// Capabilities of the replica
            /// </summary>
            public List<string> Capabilities { get; set; }

            /// <summary>
            /// Additional metadata for the replica
            /// </summary>
            public Dictionary<string, string> Metadata { get; set; }

            /// <summary>
            /// Performance metrics for the replica
            /// </summary>
            public Dictionary<string, double> Metrics { get; set; }

            /// <summary>
            /// Health status of the replica
            /// </summary>
            public bool IsHealthy { get; set; }

            /// <summary>
            /// Version of the replica
            /// </summary>
            public string Version { get; set; }
        }

        /// <summary>
        /// Initializes a new instance of the TarsReplicaManager class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="configuration">Configuration instance</param>
        /// <param name="dockerService">Docker service instance</param>
        /// <param name="mcpService">MCP service instance</param>
        public TarsReplicaManager(
            ILogger<TarsReplicaManager> logger,
            IConfiguration configuration,
            DockerService dockerService,
            McpService mcpService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _dockerService = dockerService ?? throw new ArgumentNullException(nameof(dockerService));
            _mcpService = mcpService ?? throw new ArgumentNullException(nameof(mcpService));

            // Get configuration values
            _replicaConfigPath = _configuration["Tars:Replicas:ConfigPath"] ?? "config/tars-replicas.json";
            _dockerComposeTemplatePath = _configuration["Tars:Replicas:DockerComposeTemplatePath"] ?? "templates/docker-compose-tars-replica.yml";
            _dockerComposeOutputDir = _configuration["Tars:Replicas:DockerComposeOutputDir"] ?? "docker/tars-replicas";

            // Create the output directory if it doesn't exist
            Directory.CreateDirectory(_dockerComposeOutputDir);

            // Load existing replicas
            LoadReplicas();
        }

        /// <summary>
        /// Loads existing replicas from the configuration file
        /// </summary>
        private void LoadReplicas()
        {
            try
            {
                if (File.Exists(_replicaConfigPath))
                {
                    var json = File.ReadAllText(_replicaConfigPath);
                    var jsonDoc = JsonDocument.Parse(json);
                    var replicas = JsonSerializer.Deserialize<List<ReplicaInfo>>(jsonDoc.RootElement.GetProperty("replicas").GetRawText());
                    if (replicas != null)
                    {
                        foreach (var replica in replicas)
                        {
                            _replicas[replica.Id] = replica;
                            if (replica.Port >= _nextReplicaPort)
                            {
                                _nextReplicaPort = replica.Port + 1;
                            }
                        }
                        _logger.LogInformation($"Loaded {_replicas.Count} replicas from configuration");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading replicas from configuration");
            }
        }

        /// <summary>
        /// Saves the current replicas to the configuration file
        /// </summary>
        private void SaveReplicas()
        {
            try
            {
                var replicasObj = new { replicas = _replicas.Values.ToList() };
                var json = JsonSerializer.Serialize(replicasObj, new JsonSerializerOptions { WriteIndented = true });
                Directory.CreateDirectory(Path.GetDirectoryName(_replicaConfigPath));
                File.WriteAllText(_replicaConfigPath, json);
                _logger.LogInformation($"Saved {_replicas.Count} replicas to configuration");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving replicas to configuration");
            }
        }

        /// <summary>
        /// Creates a new replica
        /// </summary>
        /// <param name="name">Name of the replica</param>
        /// <param name="role">Role of the replica</param>
        /// <param name="capabilities">Capabilities of the replica</param>
        /// <param name="metadata">Additional metadata for the replica</param>
        /// <returns>The created replica info</returns>
        public async Task<ReplicaInfo> CreateReplicaAsync(string name, string role, List<string> capabilities = null, Dictionary<string, string> metadata = null)
        {
            try
            {
                // Check if Docker is running
                if (!await _dockerService.IsDockerRunning())
                {
                    throw new InvalidOperationException("Docker is not running. Please start Docker first.");
                }

                // Create a unique ID for the replica
                var id = Guid.NewGuid().ToString("N");
                var port = _nextReplicaPort++;
                var containerName = $"tars-replica-{id.Substring(0, 8)}";
                var dockerComposePath = Path.Combine(_dockerComposeOutputDir, $"docker-compose-{id.Substring(0, 8)}.yml");

                // Create the replica info
                var replica = new ReplicaInfo
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
                    Metadata = metadata ?? new Dictionary<string, string>(),
                    Metrics = new Dictionary<string, double>(),
                    IsHealthy = false,
                    Version = typeof(TarsReplicaManager).Assembly.GetName().Version.ToString()
                };

                // Add the replica to the dictionary
                _replicas[id] = replica;
                SaveReplicas();

                // Simulate Docker Compose file creation and container start
                replica.Status = "running";
                replica.IsHealthy = true;
                SaveReplicas();
                _logger.LogInformation($"Created and started replica {name} (ID: {id}) on port {port} (simulated)");

                return replica;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error creating replica {name}");
                throw;
            }
        }

        /// <summary>
        /// Creates a Docker Compose file for a replica
        /// </summary>
        /// <param name="replica">The replica info</param>
        private async Task CreateDockerComposeFileAsync(ReplicaInfo replica)
        {
            try
            {
                // Read the template
                var template = await File.ReadAllTextAsync(_dockerComposeTemplatePath);

                // Replace placeholders
                var content = template
                    .Replace("{{REPLICA_ID}}", replica.Id)
                    .Replace("{{REPLICA_NAME}}", replica.Name)
                    .Replace("{{REPLICA_ROLE}}", replica.Role)
                    .Replace("{{REPLICA_PORT}}", replica.Port.ToString())
                    .Replace("{{CONTAINER_NAME}}", replica.ContainerName);

                // Write the Docker Compose file
                Directory.CreateDirectory(Path.GetDirectoryName(replica.DockerComposePath));
                await File.WriteAllTextAsync(replica.DockerComposePath, content);
                _logger.LogInformation($"Created Docker Compose file for replica {replica.Name} at {replica.DockerComposePath}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error creating Docker Compose file for replica {replica.Name}");
                throw;
            }
        }

        /// <summary>
        /// Starts a replica
        /// </summary>
        /// <param name="replicaId">ID of the replica to start</param>
        /// <returns>True if the replica was started successfully, false otherwise</returns>
        public async Task<bool> StartReplicaAsync(string replicaId)
        {
            try
            {
                if (!_replicas.TryGetValue(replicaId, out var replica))
                {
                    _logger.LogError($"Replica with ID {replicaId} not found");
                    return false;
                }

                // Check if Docker is running
                if (!await _dockerService.IsDockerRunning())
                {
                    _logger.LogError("Docker is not running. Please start Docker first.");
                    return false;
                }

                // Simulate container start
                replica.Status = "running";
                replica.LastActiveAt = DateTime.UtcNow;
                SaveReplicas();
                _logger.LogInformation($"Started replica {replica.Name} (ID: {replicaId}) (simulated)");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error starting replica with ID {replicaId}");
                return false;
            }
        }

        /// <summary>
        /// Stops a replica
        /// </summary>
        /// <param name="replicaId">ID of the replica to stop</param>
        /// <returns>True if the replica was stopped successfully, false otherwise</returns>
        public async Task<bool> StopReplicaAsync(string replicaId)
        {
            try
            {
                if (!_replicas.TryGetValue(replicaId, out var replica))
                {
                    _logger.LogError($"Replica with ID {replicaId} not found");
                    return false;
                }

                // Check if Docker is running
                if (!await _dockerService.IsDockerRunning())
                {
                    _logger.LogError("Docker is not running. Please start Docker first.");
                    return false;
                }

                // Simulate container stop
                replica.Status = "stopped";
                SaveReplicas();
                _logger.LogInformation($"Stopped replica {replica.Name} (ID: {replicaId}) (simulated)");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error stopping replica with ID {replicaId}");
                return false;
            }
        }

        /// <summary>
        /// Removes a replica
        /// </summary>
        /// <param name="replicaId">ID of the replica to remove</param>
        /// <returns>True if the replica was removed successfully, false otherwise</returns>
        public async Task<bool> RemoveReplicaAsync(string replicaId)
        {
            try
            {
                if (!_replicas.TryGetValue(replicaId, out var replica))
                {
                    _logger.LogError($"Replica with ID {replicaId} not found");
                    return false;
                }

                // Check if Docker is running
                if (!await _dockerService.IsDockerRunning())
                {
                    _logger.LogError("Docker is not running. Please start Docker first.");
                    return false;
                }

                // Simulate container stop and removal
                _logger.LogInformation($"Stopped and removed container for replica {replica.Name} (ID: {replicaId}) (simulated)");

                // Remove the replica from the dictionary
                _replicas.Remove(replicaId);
                SaveReplicas();
                _logger.LogInformation($"Removed replica {replica.Name} (ID: {replicaId})");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error removing replica with ID {replicaId}");
                return false;
            }
        }

        /// <summary>
        /// Gets all replicas
        /// </summary>
        /// <returns>List of all replicas</returns>
        public List<ReplicaInfo> GetAllReplicas()
        {
            return _replicas.Values.ToList();
        }

        /// <summary>
        /// Gets a replica by ID
        /// </summary>
        /// <param name="replicaId">ID of the replica to get</param>
        /// <returns>The replica info, or null if not found</returns>
        public ReplicaInfo GetReplica(string replicaId)
        {
            return _replicas.TryGetValue(replicaId, out var replica) ? replica : null;
        }

        /// <summary>
        /// Gets replicas by role
        /// </summary>
        /// <param name="role">Role to filter by</param>
        /// <returns>List of replicas with the specified role</returns>
        public List<ReplicaInfo> GetReplicasByRole(string role)
        {
            return _replicas.Values.Where(r => r.Role == role).ToList();
        }

        /// <summary>
        /// Updates the status of all replicas
        /// </summary>
        public async Task UpdateReplicaStatusesAsync()
        {
            try
            {
                // Check if Docker is running
                if (!await _dockerService.IsDockerRunning())
                {
                    _logger.LogWarning("Docker is not running. Cannot update replica statuses.");
                    return;
                }

                foreach (var replica in _replicas.Values)
                {
                    try
                    {
                        // Simulate container status check
                        replica.Status = "running";
                        replica.IsHealthy = true;

                        // Add some simulated metrics
                        replica.Metrics["cpu_usage"] = 0.2;
                        replica.Metrics["memory_usage"] = 128.5;
                        replica.Metrics["requests_processed"] = 42;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, $"Error updating status for replica {replica.Name} (ID: {replica.Id})");
                    }
                }

                SaveReplicas();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating replica statuses");
            }
        }

        /// <summary>
        /// Sends a request to a replica
        /// </summary>
        /// <param name="replicaId">ID of the replica to send the request to</param>
        /// <param name="request">The request to send</param>
        /// <returns>The response from the replica</returns>
        public async Task<JsonElement> SendRequestToReplicaAsync(string replicaId, JsonElement request)
        {
            try
            {
                if (!_replicas.TryGetValue(replicaId, out var replica))
                {
                    throw new ArgumentException($"Replica with ID {replicaId} not found");
                }

                if (replica.Status != "running" || !replica.IsHealthy)
                {
                    throw new InvalidOperationException($"Replica {replica.Name} (ID: {replicaId}) is not running or not healthy");
                }

                // Update the last active time
                replica.LastActiveAt = DateTime.UtcNow;
                SaveReplicas();

                // Extract action from the request
                string action = "code";
                if (request.TryGetProperty("action", out var actionElement))
                {
                    action = actionElement.GetString();
                }

                // Simulate sending request to replica
                _logger.LogInformation($"Sending {action} request to replica {replica.Name} (ID: {replicaId}) (simulated)");

                // Create a simulated response based on the request
                var responseObj = new
                {
                    success = true,
                    message = $"Request processed by {replica.Name}",
                    timestamp = DateTime.UtcNow.ToString("o"),
                    request_id = Guid.NewGuid().ToString(),
                    replica_id = replicaId,
                    replica_name = replica.Name,
                    replica_role = replica.Role
                };

                var responseJson = JsonSerializer.Serialize(responseObj);
                return JsonSerializer.Deserialize<JsonElement>(responseJson);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error sending request to replica with ID {replicaId}");
                throw;
            }
        }

        /// <summary>
        /// Starts all replicas
        /// </summary>
        /// <returns>True if all replicas were started successfully, false otherwise</returns>
        public async Task<bool> StartAllReplicasAsync()
        {
            var success = true;
            foreach (var replicaId in _replicas.Keys)
            {
                if (!await StartReplicaAsync(replicaId))
                {
                    success = false;
                }
            }
            return success;
        }

        /// <summary>
        /// Stops all replicas
        /// </summary>
        /// <returns>True if all replicas were stopped successfully, false otherwise</returns>
        public async Task<bool> StopAllReplicasAsync()
        {
            var success = true;
            foreach (var replicaId in _replicas.Keys)
            {
                if (!await StopReplicaAsync(replicaId))
                {
                    success = false;
                }
            }
            return success;
        }

        /// <summary>
        /// Creates a set of replicas for self-coding
        /// </summary>
        /// <returns>True if all replicas were created successfully, false otherwise</returns>
        public async Task<bool> CreateSelfCodingReplicasAsync()
        {
            try
            {
                // Create analyzer replica
                var analyzerReplica = await CreateReplicaAsync(
                    "CodeAnalyzer",
                    "analyzer",
                    new List<string> { "analyze_code", "detect_issues", "suggest_improvements" });

                // Create generator replica
                var generatorReplica = await CreateReplicaAsync(
                    "CodeGenerator",
                    "generator",
                    new List<string> { "generate_code", "refactor_code", "optimize_code" });

                // Create tester replica
                var testerReplica = await CreateReplicaAsync(
                    "TestGenerator",
                    "tester",
                    new List<string> { "generate_tests", "run_tests", "analyze_test_results" });

                // Create coordinator replica
                var coordinatorReplica = await CreateReplicaAsync(
                    "Coordinator",
                    "coordinator",
                    new List<string> { "coordinate_workflow", "prioritize_tasks", "track_progress" });

                return analyzerReplica != null && generatorReplica != null && testerReplica != null && coordinatorReplica != null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating self-coding replicas");
                return false;
            }
        }
    }
}
