using Microsoft.Extensions.Logging;
using Microsoft.FSharp.Core;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TarsEngine.SelfImprovement;

namespace TarsCli.Services;

/// <summary>
/// Service for autonomous improvement of the codebase
/// </summary>
public class AutonomousImprovementService
{
    private readonly ILogger<AutonomousImprovementService> _logger;
    private readonly ConsoleService _consoleService;
    private readonly OllamaService _ollamaService;
    private CancellationTokenSource? _cancellationTokenSource;
    private Task? _currentWorkflowTask;

    /// <summary>
    /// Initializes a new instance of the <see cref="AutonomousImprovementService"/> class.
    /// </summary>
    public AutonomousImprovementService(
        ILogger<AutonomousImprovementService> logger,
        ConsoleService consoleService,
        OllamaService ollamaService)
    {
        _logger = logger;
        _consoleService = consoleService;
        _ollamaService = ollamaService;
    }

    /// <summary>
    /// Starts the autonomous improvement workflow
    /// </summary>
    /// <param name="name">The name of the workflow</param>
    /// <param name="targetDirectories">The target directories to improve</param>
    /// <param name="maxDurationMinutes">The maximum duration of the workflow in minutes</param>
    /// <param name="maxImprovements">The maximum number of improvements to apply</param>
    /// <returns>True if the workflow was started successfully</returns>
    public async Task<bool> StartWorkflowAsync(string name, List<string> targetDirectories, int maxDurationMinutes, int maxImprovements)
    {
        try
        {
            // Check if a workflow is already running
            if (_currentWorkflowTask != null && !_currentWorkflowTask.IsCompleted)
            {
                _consoleService.WriteError("A workflow is already running");
                return false;
            }

            // Create a cancellation token source
            _cancellationTokenSource = new CancellationTokenSource();

            // Start the workflow in a background task
            _currentWorkflowTask = Task.Run(async () =>
            {
                try
                {
                    await RunWorkflowAsync(name, targetDirectories, maxDurationMinutes, maxImprovements, _cancellationTokenSource.Token);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error running workflow");
                }
            });

            _consoleService.WriteSuccess($"Started workflow: {name}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting workflow");
            _consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Stops the current workflow
    /// </summary>
    /// <returns>True if the workflow was stopped successfully</returns>
    public bool StopWorkflow()
    {
        try
        {
            // Check if a workflow is running
            if (_currentWorkflowTask == null || _currentWorkflowTask.IsCompleted || _cancellationTokenSource == null)
            {
                _consoleService.WriteWarning("No workflow is currently running");
                return false;
            }

            // Cancel the workflow
            _cancellationTokenSource.Cancel();
            _consoleService.WriteSuccess("Workflow stop requested");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping workflow");
            _consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Gets the status of the current workflow
    /// </summary>
    /// <returns>The workflow status</returns>
    public async Task<WorkflowState?> GetWorkflowStatusAsync()
    {
        try
        {
            // Try to load the workflow state
            var stateOption = await FSharpAsync.StartAsTask(
                WorkflowState.tryLoad(WorkflowState.defaultStatePath),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            if (stateOption.IsNone())
            {
                return null;
            }

            return stateOption.Value;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting workflow status");
            return null;
        }
    }

    /// <summary>
    /// Runs the autonomous improvement workflow
    /// </summary>
    private async Task RunWorkflowAsync(string name, List<string> targetDirectories, int maxDurationMinutes, int maxImprovements, CancellationToken cancellationToken)
    {
        try
        {
            _consoleService.WriteHeader("TARS Autonomous Improvement Workflow");
            _consoleService.WriteInfo($"Name: {name}");
            _consoleService.WriteInfo($"Target Directories: {string.Join(", ", targetDirectories)}");
            _consoleService.WriteInfo($"Maximum Duration: {maxDurationMinutes} minutes");
            _consoleService.WriteInfo($"Maximum Improvements: {maxImprovements}");

            // Create the step handlers
            var handlers = new List<FSharpFunc<WorkflowState, Task<FSharpResult<Microsoft.FSharp.Collections.FSharpMap<string, string>, string>>>>
            {
                // Knowledge extraction step
                FSharpFunc<WorkflowState, Task<FSharpResult<Microsoft.FSharp.Collections.FSharpMap<string, string>, string>>>.FromConverter(
                    state => KnowledgeExtractionStep.getHandler(_logger, 10)(state)),
                
                // Code analysis step
                FSharpFunc<WorkflowState, Task<FSharpResult<Microsoft.FSharp.Collections.FSharpMap<string, string>, string>>>.FromConverter(
                    state => CodeAnalysisStep.getHandler(_logger)(state)),
                
                // Improvement application step
                FSharpFunc<WorkflowState, Task<FSharpResult<Microsoft.FSharp.Collections.FSharpMap<string, string>, string>>>.FromConverter(
                    state => ImprovementApplicationStep.getHandler(_logger, maxImprovements)(state)),
                
                // Feedback collection step
                FSharpFunc<WorkflowState, Task<FSharpResult<Microsoft.FSharp.Collections.FSharpMap<string, string>, string>>>.FromConverter(
                    state => FeedbackCollectionStep.getHandler(_logger)(state)),
                
                // Reporting step
                FSharpFunc<WorkflowState, Task<FSharpResult<Microsoft.FSharp.Collections.FSharpMap<string, string>, string>>>.FromConverter(
                    state => ReportingStep.getHandler(_logger)(state))
            };

            // Create and execute the workflow
            var result = await FSharpAsync.StartAsTask(
                WorkflowEngine.createAndExecuteWorkflow(
                    _logger,
                    name,
                    targetDirectories.ToArray(),
                    maxDurationMinutes,
                    handlers.ToArray()),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Check if the workflow was cancelled
            if (cancellationToken.IsCancellationRequested)
            {
                _consoleService.WriteWarning("Workflow was cancelled");
                return;
            }

            // Check the workflow status
            if (result.Status == StepStatus.Completed)
            {
                _consoleService.WriteSuccess("Workflow completed successfully");

                // Check if a report was generated
                var reportStep = result.Steps.LastOrDefault(s => s.Name.Contains("Report") && s.Status == StepStatus.Completed);
                if (reportStep != null && reportStep.Data.ContainsKey("report_path"))
                {
                    var reportPath = reportStep.Data["report_path"];
                    _consoleService.WriteInfo($"Report generated: {reportPath}");

                    // Display the report
                    if (File.Exists(reportPath))
                    {
                        var report = await File.ReadAllTextAsync(reportPath);
                        _consoleService.WriteInfo("Report:");
                        _consoleService.WriteInfo(report);
                    }
                }
            }
            else if (result.Status == StepStatus.Failed)
            {
                _consoleService.WriteError("Workflow failed");

                // Find the failed step
                var failedStep = result.Steps.FirstOrDefault(s => s.Status == StepStatus.Failed);
                if (failedStep != null && failedStep.ErrorMessage.HasValue)
                {
                    _consoleService.WriteError($"Failed step: {failedStep.Name}");
                    _consoleService.WriteError($"Error: {failedStep.ErrorMessage.Value}");
                }
            }
            else
            {
                _consoleService.WriteWarning($"Workflow ended with status: {result.Status}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running workflow");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }
}
