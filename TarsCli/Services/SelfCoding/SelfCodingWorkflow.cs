using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.Workflow;

namespace TarsCli.Services.SelfCoding
{
    /// <summary>
    /// Service for managing the self-coding workflow
    /// </summary>
    public class SelfCodingWorkflow
    {
        private readonly ILogger<SelfCodingWorkflow> _logger;
        private readonly IWorkflowCoordinator _workflowCoordinator;
        private readonly ProgressTracker _progressTracker;
        private readonly FileProcessor _fileProcessor;
        private readonly AnalysisProcessor _analysisProcessor;
        private readonly CodeGenerationProcessor _codeGenerationProcessor;
        private readonly TestProcessor _testProcessor;

        private string _currentWorkflowId;
        private WorkflowState _workflowState = new WorkflowState();
        private CancellationTokenSource _cancellationTokenSource;

        /// <summary>
        /// Initializes a new instance of the SelfCodingWorkflow class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="workflowCoordinator">Workflow coordinator</param>
        /// <param name="progressTracker">Progress tracker</param>
        /// <param name="fileProcessor">File processor</param>
        /// <param name="analysisProcessor">Analysis processor</param>
        /// <param name="codeGenerationProcessor">Code generation processor</param>
        /// <param name="testProcessor">Test processor</param>
        public SelfCodingWorkflow(
            ILogger<SelfCodingWorkflow> logger,
            IWorkflowCoordinator workflowCoordinator,
            ProgressTracker progressTracker,
            FileProcessor fileProcessor,
            AnalysisProcessor analysisProcessor,
            CodeGenerationProcessor codeGenerationProcessor,
            TestProcessor testProcessor)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _workflowCoordinator = workflowCoordinator ?? throw new ArgumentNullException(nameof(workflowCoordinator));
            _progressTracker = progressTracker ?? throw new ArgumentNullException(nameof(progressTracker));
            _fileProcessor = fileProcessor ?? throw new ArgumentNullException(nameof(fileProcessor));
            _analysisProcessor = analysisProcessor ?? throw new ArgumentNullException(nameof(analysisProcessor));
            _codeGenerationProcessor = codeGenerationProcessor ?? throw new ArgumentNullException(nameof(codeGenerationProcessor));
            _testProcessor = testProcessor ?? throw new ArgumentNullException(nameof(testProcessor));
        }

        /// <summary>
        /// Starts the self-coding workflow
        /// </summary>
        /// <param name="targetDirectories">Target directories to improve</param>
        /// <returns>True if the workflow was started successfully, false otherwise</returns>
        public async Task<bool> StartWorkflowAsync(List<string> targetDirectories)
        {
            _logger.LogInformation($"Starting self-coding workflow for directories: {string.Join(", ", targetDirectories)}");

            try
            {
                // Check if a workflow is already running
                if (!string.IsNullOrEmpty(_currentWorkflowId))
                {
                    var workflow = await _workflowCoordinator.GetWorkflowStatusAsync(_currentWorkflowId);
                    if (workflow.Status == WorkflowStatus.Running || workflow.Status == WorkflowStatus.Paused)
                    {
                        _logger.LogWarning($"A workflow is already running with ID {_currentWorkflowId}");
                        return false;
                    }
                }

                // Create a new cancellation token source
                _cancellationTokenSource = new CancellationTokenSource();

                // Create workflow parameters
                var parameters = new Dictionary<string, object>
                {
                    ["TargetDirectory"] = targetDirectories.First(),
                    ["FilePatterns"] = new[] { "*.cs", "*.fs" },
                    ["MaxFiles"] = 10
                };

                // Start the workflow
                var workflow = await _workflowCoordinator.StartWorkflowAsync("SelfCoding", parameters);
                _currentWorkflowId = workflow.Id;

                // Initialize the workflow state
                _workflowState = new WorkflowState
                {
                    Status = WorkflowStatus.Running.ToString(),
                    CurrentStage = workflow.CurrentState,
                    StartTime = workflow.StartTime,
                    FilesToProcess = new List<string>(),
                    ProcessedFiles = new List<string>(),
                    FailedFiles = new List<string>(),
                    Statistics = new Dictionary<string, string>()
                };

                // Start monitoring the workflow
                _ = Task.Run(() => MonitorWorkflowAsync(_cancellationTokenSource.Token));

                _logger.LogInformation($"Self-coding workflow started with ID {_currentWorkflowId}");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting self-coding workflow");
                _workflowState.Status = WorkflowStatus.Failed.ToString();
                _workflowState.ErrorMessage = ex.Message;
                return false;
            }
        }

        /// <summary>
        /// Stops the self-coding workflow
        /// </summary>
        /// <returns>True if the workflow was stopped successfully, false otherwise</returns>
        public async Task<bool> StopWorkflowAsync()
        {
            _logger.LogInformation("Stopping self-coding workflow");

            try
            {
                // Check if a workflow is running
                if (string.IsNullOrEmpty(_currentWorkflowId))
                {
                    _logger.LogWarning("No workflow is currently running");
                    return false;
                }

                // Stop the workflow
                var result = await _workflowCoordinator.StopWorkflowAsync(_currentWorkflowId);
                if (result)
                {
                    // Cancel the monitoring task
                    _cancellationTokenSource?.Cancel();

                    // Update the workflow state
                    _workflowState.Status = WorkflowStatus.Cancelled.ToString();
                    _workflowState.EndTime = DateTime.UtcNow;

                    _logger.LogInformation($"Self-coding workflow {_currentWorkflowId} stopped");
                    return true;
                }
                else
                {
                    _logger.LogWarning($"Failed to stop self-coding workflow {_currentWorkflowId}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping self-coding workflow");
                return false;
            }
        }

        /// <summary>
        /// Gets the current state of the self-coding workflow
        /// </summary>
        /// <returns>Workflow state</returns>
        public WorkflowState GetWorkflowState()
        {
            return _workflowState;
        }

        /// <summary>
        /// Monitors the self-coding workflow
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Task representing the asynchronous operation</returns>
        private async Task MonitorWorkflowAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation($"Starting to monitor workflow {_currentWorkflowId}");

            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    // Get the workflow status
                    var workflow = await _workflowCoordinator.GetWorkflowStatusAsync(_currentWorkflowId);

                    // Update the workflow state
                    _workflowState.Status = workflow.Status.ToString();
                    _workflowState.CurrentStage = workflow.CurrentState;
                    _workflowState.EndTime = workflow.EndTime;

                    // Update statistics
                    if (workflow.Results.TryGetValue("SelectedFiles", out var selectedFilesObj) && selectedFilesObj is List<string> selectedFiles)
                    {
                        _workflowState.FilesToProcess = selectedFiles;
                        _workflowState.Statistics["FilesToProcess"] = selectedFiles.Count.ToString();
                    }

                    if (workflow.Results.TryGetValue("AppliedChanges", out var appliedChangesObj) && appliedChangesObj is List<string> appliedChanges)
                    {
                        _workflowState.ProcessedFiles = appliedChanges;
                        _workflowState.Statistics["ProcessedFiles"] = appliedChanges.Count.ToString();
                    }

                    // Check if the workflow has completed
                    if (workflow.Status == WorkflowStatus.Completed || workflow.Status == WorkflowStatus.Failed || workflow.Status == WorkflowStatus.Cancelled)
                    {
                        _logger.LogInformation($"Workflow {_currentWorkflowId} has {workflow.Status}");
                        break;
                    }

                    // Wait before checking again
                    await Task.Delay(5000, cancellationToken);
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation($"Monitoring of workflow {_currentWorkflowId} was cancelled");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error monitoring workflow {_currentWorkflowId}");
                _workflowState.Status = WorkflowStatus.Failed.ToString();
                _workflowState.ErrorMessage = ex.Message;
            }
        }
    }

    /// <summary>
    /// State of the self-coding workflow
    /// </summary>
    public class WorkflowState
    {
        /// <summary>
        /// Status of the workflow
        /// </summary>
        public string Status { get; set; } = "NotStarted";

        /// <summary>
        /// Current stage of the workflow
        /// </summary>
        public string CurrentStage { get; set; }

        /// <summary>
        /// Current file being processed
        /// </summary>
        public string CurrentFile { get; set; }

        /// <summary>
        /// Start time of the workflow
        /// </summary>
        public DateTime StartTime { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// End time of the workflow
        /// </summary>
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// List of files to process
        /// </summary>
        public List<string> FilesToProcess { get; set; } = new List<string>();

        /// <summary>
        /// List of processed files
        /// </summary>
        public List<string> ProcessedFiles { get; set; } = new List<string>();

        /// <summary>
        /// List of failed files
        /// </summary>
        public List<string> FailedFiles { get; set; } = new List<string>();

        /// <summary>
        /// Statistics of the workflow
        /// </summary>
        public Dictionary<string, string> Statistics { get; set; } = new Dictionary<string, string>();

        /// <summary>
        /// Error message if the workflow failed
        /// </summary>
        public string ErrorMessage { get; set; }
    }
}
