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
    /// Service for autonomous self-improvement of TARS
    /// </summary>
    public class AutoImprovementService
    {
        private readonly ILogger<AutoImprovementService> _logger;
        private readonly IConfiguration _configuration;
        private readonly SelfImprovementService _selfImprovementService;
        private readonly OllamaService _ollamaService;
        private readonly string _stateFilePath;
        private readonly string _docsDir;
        private readonly string _chatsDir;
        private CancellationTokenSource? _cancellationTokenSource;
        private bool _isRunning = false;
        private DateTime _startTime;
        private TimeSpan _timeLimit;
        private AutoImprovementState _state;

        public AutoImprovementService(
            ILogger<AutoImprovementService> logger,
            IConfiguration configuration,
            SelfImprovementService selfImprovementService,
            OllamaService ollamaService)
        {
            _logger = logger;
            _configuration = configuration;
            _selfImprovementService = selfImprovementService;
            _ollamaService = ollamaService;

            // Get project root directory
            var projectRoot = _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory();

            // Set directories for improvement
            _docsDir = Path.Combine(projectRoot, "TarsCli", "doc");
            _chatsDir = Path.Combine(projectRoot, "docs", "Explorations", "v1", "Chats");

            // Set state file path
            _stateFilePath = Path.Combine(projectRoot, "auto_improvement_state.json");

            // Initialize state
            _state = LoadState() ?? new AutoImprovementState();
        }

        /// <summary>
        /// Start autonomous improvement with a time limit
        /// </summary>
        /// <param name="timeLimit">Time limit in minutes</param>
        /// <param name="model">Model to use for improvements</param>
        /// <returns>True if started successfully, false otherwise</returns>
        public async Task<bool> StartAsync(int timeLimit, string model)
        {
            if (_isRunning)
            {
                _logger.LogWarning("Auto-improvement is already running");
                Console.WriteLine("Auto-improvement is already running. Use --status to check progress.");
                return false;
            }

            try
            {
                // Check if directories exist
                if (!Directory.Exists(_docsDir))
                {
                    _logger.LogError($"Documentation directory not found: {_docsDir}");
                    Console.WriteLine($"Error: Documentation directory not found: {_docsDir}");
                    return false;
                }

                if (!Directory.Exists(_chatsDir))
                {
                    _logger.LogError($"Chats directory not found: {_chatsDir}");
                    Console.WriteLine($"Error: Chats directory not found: {_chatsDir}");
                    return false;
                }

                // Check if model is available
                Console.WriteLine($"Checking if model {model} is available...");
                var isModelAvailable = await _ollamaService.IsModelAvailable(model);
                if (!isModelAvailable)
                {
                    _logger.LogError($"Model {model} is not available");
                    Console.WriteLine($"Error: Model {model} is not available. Please check available models with 'ollama list'.");
                    return false;
                }

                Console.WriteLine($"Model {model} is available.");

                // Set up cancellation token
                _cancellationTokenSource = new CancellationTokenSource();

                // Set time limit
                _timeLimit = TimeSpan.FromMinutes(timeLimit);
                _startTime = DateTime.Now;

                // Set running flag
                _isRunning = true;

                // Start improvement task
                _ = Task.Run(() => RunImprovementAsync(model, _cancellationTokenSource.Token));

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting auto-improvement");
                return false;
            }
        }

        /// <summary>
        /// Stop autonomous improvement
        /// </summary>
        /// <returns>True if stopped successfully, false otherwise</returns>
        public bool Stop()
        {
            if (!_isRunning)
            {
                _logger.LogWarning("Auto-improvement is not running");
                return false;
            }

            try
            {
                // Signal cancellation
                _cancellationTokenSource?.Cancel();

                // Set running flag
                _isRunning = false;

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping auto-improvement");
                return false;
            }
        }

        /// <summary>
        /// Get the current status of autonomous improvement
        /// </summary>
        /// <returns>Status information</returns>
        public AutoImprovementStatus GetStatus()
        {
            return new AutoImprovementStatus
            {
                IsRunning = _isRunning,
                StartTime = _startTime,
                TimeLimit = _timeLimit,
                ElapsedTime = _isRunning ? DateTime.Now - _startTime : TimeSpan.Zero,
                RemainingTime = _isRunning ? _timeLimit - (DateTime.Now - _startTime) : TimeSpan.Zero,
                FilesProcessed = _state.ProcessedFiles.Count,
                FilesRemaining = _state.PendingFiles.Count,
                CurrentFile = _state.CurrentFile,
                LastImprovedFile = _state.LastImprovedFile,
                TotalImprovements = _state.TotalImprovements
            };
        }

        /// <summary>
        /// Run the autonomous improvement process
        /// </summary>
        /// <param name="model">Model to use for improvements</param>
        /// <param name="cancellationToken">Cancellation token</param>
        private async Task RunImprovementAsync(string model, CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation($"Starting autonomous improvement with time limit of {_timeLimit.TotalMinutes} minutes");

                // Initialize file lists if needed
                if (_state.PendingFiles.Count == 0 && _state.ProcessedFiles.Count == 0)
                {
                    await InitializeFileListsAsync();
                }

                // Process files until time limit or cancellation
                while (!cancellationToken.IsCancellationRequested &&
                       DateTime.Now - _startTime < _timeLimit &&
                       _state.PendingFiles.Count > 0)
                {
                    // Get next file to process
                    var filePath = GetNextFileToProcess();
                    if (string.IsNullOrEmpty(filePath))
                    {
                        _logger.LogInformation("No more files to process");
                        break;
                    }

                    // Update state
                    _state.CurrentFile = filePath;
                    SaveState();

                    // Process file
                    _logger.LogInformation($"Processing file: {filePath}");

                    try
                    {
                        // Analyze and improve file
                        var success = await _selfImprovementService.RewriteFile(filePath, model, true);

                        if (success)
                        {
                            _logger.LogInformation($"Successfully improved file: {filePath}");
                            _state.LastImprovedFile = filePath;
                            _state.TotalImprovements++;
                        }
                        else
                        {
                            _logger.LogWarning($"Failed to improve file: {filePath}");
                        }

                        // Mark file as processed
                        _state.ProcessedFiles.Add(filePath);
                        _state.PendingFiles.Remove(filePath);
                        _state.CurrentFile = null;
                        SaveState();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, $"Error processing file: {filePath}");

                        // Mark file as processed to avoid getting stuck
                        _state.ProcessedFiles.Add(filePath);
                        _state.PendingFiles.Remove(filePath);
                        _state.CurrentFile = null;
                        SaveState();
                    }

                    // Check if we should stop
                    if (DateTime.Now - _startTime >= _timeLimit)
                    {
                        _logger.LogInformation("Time limit reached, stopping autonomous improvement");
                        break;
                    }

                    // Add a small delay to avoid overloading the system
                    await Task.Delay(1000, cancellationToken);
                }

                _logger.LogInformation("Autonomous improvement completed");
                _logger.LogInformation($"Processed {_state.ProcessedFiles.Count} files");
                _logger.LogInformation($"Made {_state.TotalImprovements} improvements");
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("Autonomous improvement was cancelled");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during autonomous improvement");
            }
            finally
            {
                // Reset running state
                _isRunning = false;
                SaveState();
            }
        }

        /// <summary>
        /// Initialize the lists of files to process
        /// </summary>
        private async Task InitializeFileListsAsync()
        {
            _logger.LogInformation("Initializing file lists for autonomous improvement");

            // Clear existing lists
            _state.PendingFiles.Clear();
            _state.ProcessedFiles.Clear();

            // Get files from docs directory
            if (Directory.Exists(_docsDir))
            {
                var docsFiles = Directory.GetFiles(_docsDir, "*.md", SearchOption.AllDirectories)
                    .ToList();

                _logger.LogInformation($"Found {docsFiles.Count} documentation files");
                _state.PendingFiles.AddRange(docsFiles);
            }
            else
            {
                _logger.LogWarning($"Documentation directory not found: {_docsDir}");
            }

            // Get files from chats directory
            if (Directory.Exists(_chatsDir))
            {
                var chatFiles = Directory.GetFiles(_chatsDir, "*.md", SearchOption.AllDirectories)
                    .ToList();

                // Sort chat files by quality (using file size as a proxy for now)
                chatFiles = chatFiles
                    .Select(f => new { Path = f, Info = new FileInfo(f) })
                    .OrderByDescending(f => f.Info.Length) // Larger files might have more content
                    .Select(f => f.Path)
                    .ToList();

                _logger.LogInformation($"Found {chatFiles.Count} chat files");
                _state.PendingFiles.AddRange(chatFiles);
            }
            else
            {
                _logger.LogWarning($"Chats directory not found: {_chatsDir}");
            }

            // Prioritize files
            PrioritizeFiles();

            // Save state
            SaveState();

            _logger.LogInformation($"Initialized {_state.PendingFiles.Count} files for processing");
        }

        /// <summary>
        /// Prioritize files for processing
        /// </summary>
        private void PrioritizeFiles()
        {
            // This is a simple prioritization strategy
            // In a more advanced implementation, we could use more sophisticated criteria

            // For now, we'll prioritize:
            // 1. Documentation files (they're more structured)
            // 2. Larger chat files (they might contain more useful information)

            var docFiles = _state.PendingFiles
                .Where(f => f.StartsWith(_docsDir))
                .ToList();

            var chatFiles = _state.PendingFiles
                .Where(f => f.StartsWith(_chatsDir))
                .Select(f => new { Path = f, Info = new FileInfo(f) })
                .OrderByDescending(f => f.Info.Length)
                .Select(f => f.Path)
                .ToList();

            _state.PendingFiles.Clear();
            _state.PendingFiles.AddRange(docFiles);
            _state.PendingFiles.AddRange(chatFiles);
        }

        /// <summary>
        /// Get the next file to process
        /// </summary>
        /// <returns>File path</returns>
        private string GetNextFileToProcess()
        {
            // If we have a current file, continue with it
            if (!string.IsNullOrEmpty(_state.CurrentFile) && _state.PendingFiles.Contains(_state.CurrentFile))
            {
                return _state.CurrentFile;
            }

            // Otherwise, get the next file from the pending list
            return _state.PendingFiles.FirstOrDefault() ?? string.Empty;
        }

        /// <summary>
        /// Load the state from the state file
        /// </summary>
        /// <returns>State object</returns>
        private AutoImprovementState? LoadState()
        {
            try
            {
                if (File.Exists(_stateFilePath))
                {
                    var json = File.ReadAllText(_stateFilePath);
                    return JsonSerializer.Deserialize<AutoImprovementState>(json);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading auto-improvement state");
            }

            return null;
        }

        /// <summary>
        /// Save the state to the state file
        /// </summary>
        private void SaveState()
        {
            try
            {
                var json = JsonSerializer.Serialize(_state, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(_stateFilePath, json);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving auto-improvement state");
            }
        }
    }

    /// <summary>
    /// State of the autonomous improvement process
    /// </summary>
    public class AutoImprovementState
    {
        /// <summary>
        /// Files that have been processed
        /// </summary>
        public List<string> ProcessedFiles { get; set; } = new List<string>();

        /// <summary>
        /// Files that are pending processing
        /// </summary>
        public List<string> PendingFiles { get; set; } = new List<string>();

        /// <summary>
        /// Current file being processed
        /// </summary>
        public string? CurrentFile { get; set; }

        /// <summary>
        /// Last file that was successfully improved
        /// </summary>
        public string? LastImprovedFile { get; set; }

        /// <summary>
        /// Total number of improvements made
        /// </summary>
        public int TotalImprovements { get; set; }
    }

    /// <summary>
    /// Status of the autonomous improvement process
    /// </summary>
    public class AutoImprovementStatus
    {
        /// <summary>
        /// Whether the process is running
        /// </summary>
        public bool IsRunning { get; set; }

        /// <summary>
        /// When the process started
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Time limit for the process
        /// </summary>
        public TimeSpan TimeLimit { get; set; }

        /// <summary>
        /// Time elapsed since the process started
        /// </summary>
        public TimeSpan ElapsedTime { get; set; }

        /// <summary>
        /// Time remaining before the time limit is reached
        /// </summary>
        public TimeSpan RemainingTime { get; set; }

        /// <summary>
        /// Number of files processed
        /// </summary>
        public int FilesProcessed { get; set; }

        /// <summary>
        /// Number of files remaining
        /// </summary>
        public int FilesRemaining { get; set; }

        /// <summary>
        /// Current file being processed
        /// </summary>
        public string? CurrentFile { get; set; }

        /// <summary>
        /// Last file that was successfully improved
        /// </summary>
        public string? LastImprovedFile { get; set; }

        /// <summary>
        /// Total number of improvements made
        /// </summary>
        public int TotalImprovements { get; set; }
    }
}
