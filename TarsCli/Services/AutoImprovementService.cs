using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Models;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for autonomous self-improvement of TARS
    /// </summary>
    public partial class AutoImprovementService
    {
        private readonly ILogger<AutoImprovementService> _logger;
        private readonly IConfiguration _configuration;
        private readonly SelfImprovementService _selfImprovementService;
        private readonly OllamaService _ollamaService;
        private readonly SlackIntegrationService _slackService;
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
            OllamaService ollamaService,
            SlackIntegrationService slackService)
        {
            _logger = logger;
            _configuration = configuration;
            _selfImprovementService = selfImprovementService;
            _ollamaService = ollamaService;
            _slackService = slackService;

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
            var status = new AutoImprovementStatus
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

            // Add top priority files
            var topFiles = _state.FilePriorityScores
                .OrderByDescending(kvp => kvp.Value.TotalScore)
                .Take(5)
                .ToList();

            foreach (var file in topFiles)
            {
                status.TopPriorityFiles.Add(new FilePriorityInfo
                {
                    FilePath = file.Key,
                    Score = file.Value.TotalScore,
                    Reason = GetTopScoreFactors(file.Value)
                });
            }

            // Add recent improvements
            var recentImprovements = _state.ImprovementHistory
                .OrderByDescending(i => i.Timestamp)
                .Take(5)
                .ToList();

            foreach (var improvement in recentImprovements)
            {
                status.RecentImprovements.Add(new ImprovementInfo
                {
                    FilePath = improvement.FilePath,
                    Timestamp = improvement.Timestamp,
                    Description = improvement.Description,
                    ScoreImprovement = improvement.ScoreAfter - improvement.ScoreBefore
                });
            }

            return status;
        }

        /// <summary>
        /// Get the top score factors for a file
        /// </summary>
        /// <param name="score">The file priority score</param>
        /// <returns>A string describing the top score factors</returns>
        private string GetTopScoreFactors(FilePriorityScore score)
        {
            var topFactors = score.ScoreFactors
                .OrderByDescending(kvp => kvp.Value)
                .Take(3)
                .ToList();

            return string.Join(", ", topFactors.Select(f => $"{f.Key}: {f.Value:F2}"));
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
                        // Calculate the score before improvement
                        var scoreBefore = CalculateFilePriorityScore(filePath);
                        _logger.LogInformation($"Score before improvement: {scoreBefore.TotalScore:F2}");

                        // Analyze and improve file
                        var success = await _selfImprovementService.RewriteFile(filePath, model, true);

                        if (success)
                        {
                            _logger.LogInformation($"Successfully improved file: {filePath}");

                            // Calculate the score after improvement
                            var scoreAfter = CalculateFilePriorityScore(filePath);
                            _logger.LogInformation($"Score after improvement: {scoreAfter.TotalScore:F2}");
                            _logger.LogInformation($"Score improvement: {scoreAfter.TotalScore - scoreBefore.TotalScore:F2}");

                            // Record the improvement
                            var improvement = new ImprovementRecord
                            {
                                FilePath = filePath,
                                Timestamp = DateTime.Now,
                                Description = $"Improved file using {model} model",
                                ScoreBefore = scoreBefore.TotalScore,
                                ScoreAfter = scoreAfter.TotalScore,
                                Model = model
                            };

                            _state.ImprovementHistory.Add(improvement);
                            _state.ImprovedFiles.Add(filePath);
                            _state.LastImprovedFile = filePath;
                            _state.TotalImprovements++;

                            // Update the file priority score
                            _state.FilePriorityScores[filePath] = scoreAfter;

                            // Save state
                            SaveState();

                            // Post to Slack every 5 improvements if enabled
                            if (_slackService.IsEnabled() && _state.TotalImprovements % 5 == 0)
                            {
                                var details = new System.Text.StringBuilder();
                                details.AppendLine($"TARS has made {_state.TotalImprovements} improvements so far in this session.");
                                details.AppendLine($"Processed {_state.ProcessedFiles.Count} files out of {_state.ProcessedFiles.Count + _state.PendingFiles.Count} total files.");
                                details.AppendLine($"\nLast improved file: `{Path.GetFileName(filePath)}`");

                                // Add score improvement information
                                var latestImprovement = _state.ImprovementHistory
                                    .OrderByDescending(i => i.Timestamp)
                                    .FirstOrDefault();

                                if (latestImprovement != null)
                                {
                                    var scoreImprovement = latestImprovement.ScoreAfter - latestImprovement.ScoreBefore;
                                    details.AppendLine($"Score improvement: {scoreImprovement:F2}");
                                }

                                // Add some of the most recently improved files
                                var recentlyImproved = _state.ImprovedFiles.TakeLast(Math.Min(3, _state.ImprovedFiles.Count)).ToList();
                                if (recentlyImproved.Any())
                                {
                                    details.AppendLine("\n*Recently improved files:*");
                                    foreach (var file in recentlyImproved)
                                    {
                                        details.AppendLine($"• `{Path.GetFileName(file)}`");
                                    }
                                }

                                await _slackService.PostAutoImprovementUpdateAsync(_state.TotalImprovements, details.ToString());
                            }
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

                        // Check if we should stop
                        if (DateTime.Now - _startTime >= _timeLimit)
                        {
                            _logger.LogInformation("Time limit reached, stopping autonomous improvement");
                            break;
                        }

                        // Add a small delay to avoid overloading the system
                        await Task.Delay(1000, cancellationToken);
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

                }

                _logger.LogInformation("Autonomous improvement completed");
                _logger.LogInformation($"Processed {_state.ProcessedFiles.Count} files");
                _logger.LogInformation($"Made {_state.TotalImprovements} improvements");

                // Post update to Slack if enabled
                if (_slackService.IsEnabled() && _state.TotalImprovements > 0)
                {
                    var details = new System.Text.StringBuilder();
                    details.AppendLine($"TARS has completed an autonomous improvement session and made {_state.TotalImprovements} improvements.");
                    details.AppendLine($"Processed {_state.ProcessedFiles.Count} files in {(DateTime.Now - _startTime).TotalMinutes:F1} minutes.");

                    if (!string.IsNullOrEmpty(_state.LastImprovedFile))
                    {
                        details.AppendLine($"\nLast improved file: `{Path.GetFileName(_state.LastImprovedFile)}`");
                    }

                    // Add some of the most recently improved files
                    var recentlyImproved = _state.ImprovedFiles.TakeLast(Math.Min(5, _state.ImprovedFiles.Count)).ToList();
                    if (recentlyImproved.Any())
                    {
                        details.AppendLine("\n*Recently improved files:*");
                        foreach (var file in recentlyImproved)
                        {
                            details.AppendLine($"• `{Path.GetFileName(file)}`");
                        }
                    }

                    await _slackService.PostAutoImprovementUpdateAsync(_state.TotalImprovements, details.ToString());
                }
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
            _state.FilePriorityScores.Clear();

            // Get files from docs directory
            var allFiles = new List<string>();
            if (Directory.Exists(_docsDir))
            {
                var docsFiles = Directory.GetFiles(_docsDir, "*.md", SearchOption.AllDirectories)
                    .ToList();

                _logger.LogInformation($"Found {docsFiles.Count} documentation files");
                allFiles.AddRange(docsFiles);
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

                _logger.LogInformation($"Found {chatFiles.Count} chat files");
                allFiles.AddRange(chatFiles);
            }
            else
            {
                _logger.LogWarning($"Chats directory not found: {_chatsDir}");
            }

            // Get source code files
            var projectRoot = _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory();
            var sourceCodeDirs = new[] { "TarsCli", "TarsEngine", "TarsEngine.DSL", "TarsEngine.SelfImprovement" };

            foreach (var dir in sourceCodeDirs)
            {
                var dirPath = Path.Combine(projectRoot, dir);
                if (Directory.Exists(dirPath))
                {
                    var sourceFiles = Directory.GetFiles(dirPath, "*.cs", SearchOption.AllDirectories)
                        .Concat(Directory.GetFiles(dirPath, "*.fs", SearchOption.AllDirectories))
                        .ToList();

                    _logger.LogInformation($"Found {sourceFiles.Count} source files in {dir}");
                    allFiles.AddRange(sourceFiles);
                }
            }

            // Calculate priority scores for all files
            foreach (var file in allFiles)
            {
                var score = CalculateFilePriorityScore(file);
                _state.FilePriorityScores[file] = score;
                _state.PendingFiles.Add(file);
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
            // Sort files by total priority score (descending)
            var prioritizedFiles = _state.FilePriorityScores
                .OrderByDescending(kvp => kvp.Value.TotalScore)
                .Select(kvp => kvp.Key)
                .ToList();

            // Update the pending files list with the prioritized order
            _state.PendingFiles.Clear();
            _state.PendingFiles.AddRange(prioritizedFiles);

            // Log the top 5 priority files
            var topFiles = _state.FilePriorityScores
                .OrderByDescending(kvp => kvp.Value.TotalScore)
                .Take(5)
                .ToList();

            _logger.LogInformation("Top priority files:");
            foreach (var file in topFiles)
            {
                _logger.LogInformation($"{Path.GetFileName(file.Key)}: Score = {file.Value.TotalScore:F2}");
            }
        }

        // GetTopScoreFactors method is already defined above

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
        /// Files that were successfully improved
        /// </summary>
        public List<string> ImprovedFiles { get; set; } = new List<string>();

        /// <summary>
        /// Total number of improvements made
        /// </summary>
        public int TotalImprovements { get; set; }

        /// <summary>
        /// The file priority scores
        /// </summary>
        public Dictionary<string, FilePriorityScore> FilePriorityScores { get; set; } = new Dictionary<string, FilePriorityScore>();

        /// <summary>
        /// The last time the state was updated
        /// </summary>
        public DateTime LastUpdated { get; set; } = DateTime.Now;

        /// <summary>
        /// The history of improvements made
        /// </summary>
        public List<ImprovementRecord> ImprovementHistory { get; set; } = new List<ImprovementRecord>();
    }

    /// <summary>
    /// Represents a record of an improvement made to a file
    /// </summary>
    public class ImprovementRecord
    {
        /// <summary>
        /// The path of the file that was improved
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// The time the improvement was made
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.Now;

        /// <summary>
        /// A description of the improvement
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// The score of the file before improvement
        /// </summary>
        public double ScoreBefore { get; set; }

        /// <summary>
        /// The score of the file after improvement
        /// </summary>
        public double ScoreAfter { get; set; }

        /// <summary>
        /// The model used for the improvement
        /// </summary>
        public string Model { get; set; } = string.Empty;
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

        /// <summary>
        /// The top priority files (up to 5)
        /// </summary>
        public List<FilePriorityInfo> TopPriorityFiles { get; set; } = new List<FilePriorityInfo>();

        /// <summary>
        /// The recent improvements (up to 5)
        /// </summary>
        public List<ImprovementInfo> RecentImprovements { get; set; } = new List<ImprovementInfo>();
    }

    /// <summary>
    /// Represents information about a file's priority
    /// </summary>
    public class FilePriorityInfo
    {
        /// <summary>
        /// The path of the file
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// The priority score of the file
        /// </summary>
        public double Score { get; set; }

        /// <summary>
        /// A description of why the file has this priority
        /// </summary>
        public string Reason { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents information about an improvement
    /// </summary>
    public class ImprovementInfo
    {
        /// <summary>
        /// The path of the file that was improved
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// The time the improvement was made
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// A description of the improvement
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// The score improvement
        /// </summary>
        public double ScoreImprovement { get; set; }
    }
}
