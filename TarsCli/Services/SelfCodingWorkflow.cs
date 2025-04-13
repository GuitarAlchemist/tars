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
    /// Service for managing the self-coding workflow
    /// </summary>
    public class SelfCodingWorkflow
    {
        private readonly ILogger<SelfCodingWorkflow> _logger;
        private readonly IConfiguration _configuration;
        private readonly TarsReplicaManager _replicaManager;
        private readonly DockerService _dockerService;
        private CancellationTokenSource _cancellationTokenSource;
        private Task _workflowTask;
        private bool _isRunning = false;
        private readonly string _workflowStatePath;
        private WorkflowState _currentState;

        /// <summary>
        /// Represents the state of the self-coding workflow
        /// </summary>
        public class WorkflowState
        {
            /// <summary>
            /// Current stage of the workflow
            /// </summary>
            public string CurrentStage { get; set; }

            /// <summary>
            /// List of files being processed
            /// </summary>
            public List<string> FilesToProcess { get; set; }

            /// <summary>
            /// List of files that have been processed
            /// </summary>
            public List<string> ProcessedFiles { get; set; }

            /// <summary>
            /// List of files that failed processing
            /// </summary>
            public List<string> FailedFiles { get; set; }

            /// <summary>
            /// Current file being processed
            /// </summary>
            public string CurrentFile { get; set; }

            /// <summary>
            /// Start time of the workflow
            /// </summary>
            public DateTime StartTime { get; set; }

            /// <summary>
            /// End time of the workflow
            /// </summary>
            public DateTime? EndTime { get; set; }

            /// <summary>
            /// Status of the workflow
            /// </summary>
            public string Status { get; set; }

            /// <summary>
            /// Error message if the workflow failed
            /// </summary>
            public string ErrorMessage { get; set; }

            /// <summary>
            /// Statistics about the workflow
            /// </summary>
            public Dictionary<string, int> Statistics { get; set; }

            /// <summary>
            /// Replica IDs used in the workflow
            /// </summary>
            public Dictionary<string, string> ReplicaIds { get; set; }
        }

        /// <summary>
        /// Initializes a new instance of the SelfCodingWorkflow class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="configuration">Configuration instance</param>
        /// <param name="replicaManager">Replica manager instance</param>
        /// <param name="dockerService">Docker service instance</param>
        public SelfCodingWorkflow(
            ILogger<SelfCodingWorkflow> logger,
            IConfiguration configuration,
            TarsReplicaManager replicaManager,
            DockerService dockerService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _replicaManager = replicaManager ?? throw new ArgumentNullException(nameof(replicaManager));
            _dockerService = dockerService ?? throw new ArgumentNullException(nameof(dockerService));

            // Get configuration values
            _workflowStatePath = _configuration["Tars:SelfCoding:WorkflowStatePath"] ?? "data/self-coding/workflow-state.json";

            // Load the current state if it exists
            LoadState();
        }

        /// <summary>
        /// Loads the workflow state from disk
        /// </summary>
        private void LoadState()
        {
            try
            {
                if (File.Exists(_workflowStatePath))
                {
                    var json = File.ReadAllText(_workflowStatePath);
                    _currentState = JsonSerializer.Deserialize<WorkflowState>(json);
                    _logger.LogInformation("Loaded workflow state from disk");
                }
                else
                {
                    _currentState = new WorkflowState
                    {
                        CurrentStage = "idle",
                        FilesToProcess = new List<string>(),
                        ProcessedFiles = new List<string>(),
                        FailedFiles = new List<string>(),
                        CurrentFile = null,
                        StartTime = DateTime.MinValue,
                        EndTime = null,
                        Status = "idle",
                        ErrorMessage = null,
                        Statistics = new Dictionary<string, int>(),
                        ReplicaIds = new Dictionary<string, string>()
                    };
                    _logger.LogInformation("Created new workflow state");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading workflow state");
                _currentState = new WorkflowState
                {
                    CurrentStage = "idle",
                    FilesToProcess = new List<string>(),
                    ProcessedFiles = new List<string>(),
                    FailedFiles = new List<string>(),
                    CurrentFile = null,
                    StartTime = DateTime.MinValue,
                    EndTime = null,
                    Status = "idle",
                    ErrorMessage = null,
                    Statistics = new Dictionary<string, int>(),
                    ReplicaIds = new Dictionary<string, string>()
                };
            }
        }

        /// <summary>
        /// Saves the workflow state to disk
        /// </summary>
        private void SaveState()
        {
            try
            {
                var json = JsonSerializer.Serialize(_currentState, new JsonSerializerOptions { WriteIndented = true });
                Directory.CreateDirectory(Path.GetDirectoryName(_workflowStatePath));
                File.WriteAllText(_workflowStatePath, json);
                _logger.LogInformation("Saved workflow state to disk");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving workflow state");
            }
        }

        /// <summary>
        /// Starts the self-coding workflow
        /// </summary>
        /// <param name="targetDirectories">Directories to target for self-coding</param>
        /// <returns>True if the workflow was started successfully, false otherwise</returns>
        public async Task<bool> StartWorkflowAsync(List<string> targetDirectories)
        {
            try
            {
                if (_isRunning)
                {
                    _logger.LogWarning("Self-coding workflow is already running");
                    return false;
                }

                // Check if Docker is running
                if (!await _dockerService.IsDockerRunning())
                {
                    _logger.LogError("Docker is not running. Please start Docker first.");
                    return false;
                }

                _logger.LogInformation($"Starting self-coding workflow for directories: {string.Join(", ", targetDirectories)}");

                // Create a new workflow state
                _currentState = new WorkflowState
                {
                    CurrentStage = "initializing",
                    FilesToProcess = new List<string>(),
                    ProcessedFiles = new List<string>(),
                    FailedFiles = new List<string>(),
                    CurrentFile = null,
                    StartTime = DateTime.UtcNow,
                    EndTime = null,
                    Status = "running",
                    ErrorMessage = null,
                    Statistics = new Dictionary<string, int>(),
                    ReplicaIds = new Dictionary<string, string>()
                };
                SaveState();

                // Create the replicas if they don't exist
                var replicas = _replicaManager.GetAllReplicas();
                if (replicas.Count == 0 || !replicas.Any(r => r.Role == "analyzer") || !replicas.Any(r => r.Role == "generator") || !replicas.Any(r => r.Role == "tester") || !replicas.Any(r => r.Role == "coordinator"))
                {
                    _logger.LogInformation("Creating self-coding replicas");
                    await _replicaManager.CreateSelfCodingReplicasAsync();
                }

                // Get the replica IDs
                replicas = _replicaManager.GetAllReplicas();
                var analyzerReplica = replicas.FirstOrDefault(r => r.Role == "analyzer");
                var generatorReplica = replicas.FirstOrDefault(r => r.Role == "generator");
                var testerReplica = replicas.FirstOrDefault(r => r.Role == "tester");
                var coordinatorReplica = replicas.FirstOrDefault(r => r.Role == "coordinator");

                if (analyzerReplica == null || generatorReplica == null || testerReplica == null || coordinatorReplica == null)
                {
                    _logger.LogError("Failed to create all required replicas");
                    _currentState.Status = "failed";
                    _currentState.ErrorMessage = "Failed to create all required replicas";
                    SaveState();
                    return false;
                }

                // Store the replica IDs in the workflow state
                _currentState.ReplicaIds["analyzer"] = analyzerReplica.Id;
                _currentState.ReplicaIds["generator"] = generatorReplica.Id;
                _currentState.ReplicaIds["tester"] = testerReplica.Id;
                _currentState.ReplicaIds["coordinator"] = coordinatorReplica.Id;
                SaveState();

                // Start the replicas
                await _replicaManager.StartReplicaAsync(analyzerReplica.Id);
                await _replicaManager.StartReplicaAsync(generatorReplica.Id);
                await _replicaManager.StartReplicaAsync(testerReplica.Id);
                await _replicaManager.StartReplicaAsync(coordinatorReplica.Id);

                // Get all files in the target directories
                _currentState.FilesToProcess = GetFilesInDirectories(targetDirectories);
                _currentState.Statistics["totalFiles"] = _currentState.FilesToProcess.Count;
                _currentState.CurrentStage = "scanning";
                SaveState();

                // Start the workflow task
                _cancellationTokenSource = new CancellationTokenSource();
                _workflowTask = Task.Run(() => RunWorkflowAsync(_cancellationTokenSource.Token));
                _isRunning = true;

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting self-coding workflow");
                _currentState.Status = "failed";
                _currentState.ErrorMessage = ex.Message;
                SaveState();
                return false;
            }
        }

        /// <summary>
        /// Stops the self-coding workflow
        /// </summary>
        /// <returns>True if the workflow was stopped successfully, false otherwise</returns>
        public async Task<bool> StopWorkflowAsync()
        {
            try
            {
                if (!_isRunning)
                {
                    _logger.LogWarning("Self-coding workflow is not running");
                    return false;
                }

                _logger.LogInformation("Stopping self-coding workflow");

                // Cancel the workflow task
                _cancellationTokenSource?.Cancel();
                if (_workflowTask != null)
                {
                    try
                    {
                        await _workflowTask;
                    }
                    catch (OperationCanceledException)
                    {
                        // Expected
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error waiting for workflow task to complete");
                    }
                }

                // Update the workflow state
                _currentState.Status = "stopped";
                _currentState.EndTime = DateTime.UtcNow;
                SaveState();

                _isRunning = false;
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping self-coding workflow");
                return false;
            }
        }

        /// <summary>
        /// Gets the current state of the workflow
        /// </summary>
        /// <returns>The current workflow state</returns>
        public WorkflowState GetWorkflowState()
        {
            return _currentState;
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
        /// Runs the self-coding workflow
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        private async Task RunWorkflowAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("Starting self-coding workflow");

                // Create a directory for workflow artifacts
                var workflowDir = Path.Combine("data", "self-coding", DateTime.Now.ToString("yyyyMMdd-HHmmss"));
                Directory.CreateDirectory(workflowDir);

                // Process files until the queue is empty or the task is cancelled
                _currentState.CurrentStage = "processing";
                SaveState();

                while (_currentState.FilesToProcess.Count > 0 && !cancellationToken.IsCancellationRequested)
                {
                    // Get the next file
                    var file = _currentState.FilesToProcess[0];
                    _currentState.FilesToProcess.RemoveAt(0);
                    _currentState.CurrentFile = file;
                    SaveState();

                    _logger.LogInformation($"Processing file: {file}");

                    try
                    {
                        // Analyze the file
                        var analysisResult = await AnalyzeFileAsync(file);
                        
                        // Save the analysis result
                        var analysisPath = Path.Combine(workflowDir, $"{Path.GetFileNameWithoutExtension(file)}_analysis.json");
                        File.WriteAllText(analysisPath, JsonSerializer.Serialize(analysisResult, new JsonSerializerOptions { WriteIndented = true }));

                        // Check if improvements are needed
                        if (analysisResult.TryGetProperty("needs_improvement", out var needsImprovement) && needsImprovement.GetBoolean())
                        {
                            // Generate improvements
                            var improvementResult = await GenerateImprovementsAsync(file, analysisResult);
                            
                            // Save the improvement result
                            var improvementPath = Path.Combine(workflowDir, $"{Path.GetFileNameWithoutExtension(file)}_improvements.json");
                            File.WriteAllText(improvementPath, JsonSerializer.Serialize(improvementResult, new JsonSerializerOptions { WriteIndented = true }));

                            // Apply improvements if auto-apply is enabled
                            var autoApply = _configuration.GetValue<bool>("Tars:SelfCoding:AutoApply", false);
                            if (autoApply)
                            {
                                var applyResult = await ApplyImprovementsAsync(file, improvementResult);
                                
                                // Save the apply result
                                var applyPath = Path.Combine(workflowDir, $"{Path.GetFileNameWithoutExtension(file)}_apply.json");
                                File.WriteAllText(applyPath, JsonSerializer.Serialize(applyResult, new JsonSerializerOptions { WriteIndented = true }));

                                // Run tests if improvements were applied
                                if (applyResult.TryGetProperty("success", out var applySuccess) && applySuccess.GetBoolean())
                                {
                                    var testResult = await RunTestsAsync(file);
                                    
                                    // Save the test result
                                    var testPath = Path.Combine(workflowDir, $"{Path.GetFileNameWithoutExtension(file)}_test.json");
                                    File.WriteAllText(testPath, JsonSerializer.Serialize(testResult, new JsonSerializerOptions { WriteIndented = true }));
                                }
                            }
                        }

                        _currentState.ProcessedFiles.Add(file);
                        IncrementStatistic("processedFiles");
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, $"Error processing file: {file}");
                        _currentState.FailedFiles.Add(file);
                        IncrementStatistic("failedFiles");
                    }

                    // Wait a bit to avoid overwhelming the system
                    await Task.Delay(1000, cancellationToken);
                }

                // Update the workflow state
                _currentState.CurrentStage = "completed";
                _currentState.Status = "completed";
                _currentState.EndTime = DateTime.UtcNow;
                SaveState();

                _logger.LogInformation($"Self-coding workflow completed. Processed {_currentState.ProcessedFiles.Count} files successfully, {_currentState.FailedFiles.Count} files failed.");
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("Self-coding workflow was cancelled");
                _currentState.Status = "cancelled";
                _currentState.EndTime = DateTime.UtcNow;
                SaveState();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running self-coding workflow");
                _currentState.Status = "failed";
                _currentState.ErrorMessage = ex.Message;
                _currentState.EndTime = DateTime.UtcNow;
                SaveState();
            }
            finally
            {
                _isRunning = false;
            }
        }

        /// <summary>
        /// Increments a statistic in the workflow state
        /// </summary>
        /// <param name="statistic">Name of the statistic to increment</param>
        private void IncrementStatistic(string statistic)
        {
            if (!_currentState.Statistics.ContainsKey(statistic))
            {
                _currentState.Statistics[statistic] = 0;
            }
            _currentState.Statistics[statistic]++;
            SaveState();
        }

        /// <summary>
        /// Analyzes a file using the analyzer replica
        /// </summary>
        /// <param name="filePath">Path to the file to analyze</param>
        /// <returns>Analysis result as a JsonElement</returns>
        private async Task<JsonElement> AnalyzeFileAsync(string filePath)
        {
            // Get the analyzer replica
            var analyzerReplicaId = _currentState.ReplicaIds["analyzer"];
            
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
                file_type = fileExtension
            };

            var requestJson = JsonSerializer.Serialize(requestObj);
            var request = JsonSerializer.Deserialize<JsonElement>(requestJson);

            // Send the request to the analyzer replica
            var response = await _replicaManager.SendRequestToReplicaAsync(analyzerReplicaId, request);
            IncrementStatistic("analyzedFiles");
            return response;
        }

        /// <summary>
        /// Generates improvements for a file using the generator replica
        /// </summary>
        /// <param name="filePath">Path to the file to improve</param>
        /// <param name="analysisResult">Analysis result from the analyzer replica</param>
        /// <returns>Improvement result as a JsonElement</returns>
        private async Task<JsonElement> GenerateImprovementsAsync(string filePath, JsonElement analysisResult)
        {
            // Get the generator replica
            var generatorReplicaId = _currentState.ReplicaIds["generator"];
            
            // Read the file content
            var fileContent = await File.ReadAllTextAsync(filePath);
            var fileExtension = Path.GetExtension(filePath).TrimStart('.');

            // Create the request
            var requestObj = new
            {
                action = "code",
                operation = "generate",
                file_path = filePath,
                file_content = fileContent,
                file_type = fileExtension,
                analysis = analysisResult.ToString()
            };

            var requestJson = JsonSerializer.Serialize(requestObj);
            var request = JsonSerializer.Deserialize<JsonElement>(requestJson);

            // Send the request to the generator replica
            var response = await _replicaManager.SendRequestToReplicaAsync(generatorReplicaId, request);
            IncrementStatistic("generatedImprovements");
            return response;
        }

        /// <summary>
        /// Applies improvements to a file
        /// </summary>
        /// <param name="filePath">Path to the file to improve</param>
        /// <param name="improvementResult">Improvement result from the generator replica</param>
        /// <returns>Apply result as a JsonElement</returns>
        private async Task<JsonElement> ApplyImprovementsAsync(string filePath, JsonElement improvementResult)
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
            IncrementStatistic("appliedImprovements");
            return JsonSerializer.Deserialize<JsonElement>(resultJson);
        }

        /// <summary>
        /// Runs tests for a file using the tester replica
        /// </summary>
        /// <param name="filePath">Path to the file to test</param>
        /// <returns>Test result as a JsonElement</returns>
        private async Task<JsonElement> RunTestsAsync(string filePath)
        {
            // Get the tester replica
            var testerReplicaId = _currentState.ReplicaIds["tester"];
            
            // Read the file content
            var fileContent = await File.ReadAllTextAsync(filePath);
            var fileExtension = Path.GetExtension(filePath).TrimStart('.');

            // Create the request
            var requestObj = new
            {
                action = "code",
                operation = "test",
                file_path = filePath,
                file_content = fileContent,
                file_type = fileExtension
            };

            var requestJson = JsonSerializer.Serialize(requestObj);
            var request = JsonSerializer.Deserialize<JsonElement>(requestJson);

            // Send the request to the tester replica
            var response = await _replicaManager.SendRequestToReplicaAsync(testerReplicaId, request);
            IncrementStatistic("ranTests");
            return response;
        }
    }
}
