using System.Text.Json;
using TarsCli.Models;
using TarsCli.Services.Workflow;

namespace TarsCli.Services.Mcp;

/// <summary>
/// Action handler for coordinator replica
/// </summary>
public class CoordinatorReplicaActionHandler : IMcpActionHandler
{
    private readonly ILogger<CoordinatorReplicaActionHandler> _logger;
    private readonly IWorkflowCoordinator _workflowCoordinator;
    private readonly TaskPrioritizer _taskPrioritizer;
    private readonly ProgressTracker _progressTracker;

    /// <summary>
    /// Initializes a new instance of the CoordinatorReplicaActionHandler class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="workflowCoordinator">Workflow coordinator</param>
    /// <param name="taskPrioritizer">Task prioritizer</param>
    /// <param name="progressTracker">Progress tracker</param>
    public CoordinatorReplicaActionHandler(
        ILogger<CoordinatorReplicaActionHandler> logger,
        IWorkflowCoordinator workflowCoordinator,
        TaskPrioritizer taskPrioritizer,
        ProgressTracker progressTracker)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _workflowCoordinator = workflowCoordinator ?? throw new ArgumentNullException(nameof(workflowCoordinator));
        _taskPrioritizer = taskPrioritizer ?? throw new ArgumentNullException(nameof(taskPrioritizer));
        _progressTracker = progressTracker ?? throw new ArgumentNullException(nameof(progressTracker));
    }

    /// <inheritdoc/>
    public string ActionType => "coordinate_workflow";

    /// <inheritdoc/>
    public async Task<McpActionResult> HandleActionAsync(McpAction action)
    {
        _logger.LogInformation("Handling coordinate_workflow action");

        try
        {
            // Extract parameters from the action
            var parameters = action.Parameters;
            var operation = parameters.TryGetProperty("operation", out var operationElement)
                ? operationElement.GetString()
                : "status";

            var workflowId = parameters.TryGetProperty("workflow_id", out var workflowIdElement)
                ? workflowIdElement.GetString()
                : null;

            // Handle different operations
            switch (operation)
            {
                case "start":
                    return await HandleStartWorkflowAsync(parameters, action.Id);

                case "status":
                    return await HandleGetWorkflowStatusAsync(workflowId, action.Id);

                case "stop":
                    return await HandleStopWorkflowAsync(workflowId, action.Id);

                case "transition":
                    return await HandleTransitionWorkflowAsync(parameters, action.Id);

                case "list":
                    return await HandleListWorkflowsAsync(parameters, action.Id);

                case "report":
                    return await HandleGenerateReportAsync(workflowId, action.Id);

                default:
                    return McpActionResult.CreateFailure($"Unknown operation: {operation}", action.Id);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling coordinate_workflow action");
            return McpActionResult.CreateFailure(ex.Message, action.Id);
        }
    }

    /// <summary>
    /// Handles the start workflow operation
    /// </summary>
    /// <param name="parameters">Request parameters</param>
    /// <param name="actionId">The ID of the action</param>
    /// <returns>JSON response</returns>
    private async Task<McpActionResult> HandleStartWorkflowAsync(JsonElement parameters, string actionId)
    {
        _logger.LogInformation("Starting workflow");

        // Extract parameters
        var workflowType = parameters.TryGetProperty("workflow_type", out var workflowTypeElement)
            ? workflowTypeElement.GetString()
            : "SelfCoding";

        var workflowParams = new Dictionary<string, object>();
        if (parameters.TryGetProperty("parameters", out var parametersElement) && parametersElement.ValueKind == JsonValueKind.Object)
        {
            foreach (var parameter in parametersElement.EnumerateObject())
            {
                switch (parameter.Value.ValueKind)
                {
                    case JsonValueKind.String:
                        workflowParams[parameter.Name] = parameter.Value.GetString();
                        break;
                    case JsonValueKind.Number:
                        workflowParams[parameter.Name] = parameter.Value.GetDouble();
                        break;
                    case JsonValueKind.True:
                    case JsonValueKind.False:
                        workflowParams[parameter.Name] = parameter.Value.GetBoolean();
                        break;
                    case JsonValueKind.Array:
                        var list = new List<string>();
                        foreach (var item in parameter.Value.EnumerateArray())
                        {
                            if (item.ValueKind == JsonValueKind.String)
                            {
                                list.Add(item.GetString());
                            }
                        }
                        workflowParams[parameter.Name] = list;
                        break;
                }
            }
        }

        // Start the workflow
        var workflow = await _workflowCoordinator.StartWorkflowAsync(workflowType, workflowParams);

        // Record progress
        await _progressTracker.RecordProgressAsync(workflow);

        // Convert the workflow to a JSON-friendly format
        var resultObj = new
        {
            workflow_id = workflow.Id,
            workflow_type = workflow.Type,
            current_state = workflow.CurrentState,
            status = workflow.Status.ToString().ToLowerInvariant(),
            start_time = workflow.StartTime.ToString("o")
        };

        var resultJson = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(resultObj));
        return McpActionResult.CreateSuccess(resultJson, actionId);
    }

    /// <summary>
    /// Handles the get workflow status operation
    /// </summary>
    /// <param name="workflowId">ID of the workflow</param>
    /// <param name="actionId">The ID of the action</param>
    /// <returns>JSON response</returns>
    private async Task<McpActionResult> HandleGetWorkflowStatusAsync(string workflowId, string actionId)
    {
        _logger.LogInformation($"Getting status of workflow {workflowId}");

        // Validate parameters
        if (string.IsNullOrEmpty(workflowId))
        {
            return McpActionResult.CreateFailure("Workflow ID is required", actionId);
        }

        // Get the workflow status
        var workflow = await _workflowCoordinator.GetWorkflowStatusAsync(workflowId);

        // Convert the workflow to a JSON-friendly format
        var resultObj = new
        {
            workflow_id = workflow.Id,
            workflow_type = workflow.Type,
            current_state = workflow.CurrentState,
            status = workflow.Status.ToString().ToLowerInvariant(),
            start_time = workflow.StartTime.ToString("o"),
            end_time = workflow.EndTime?.ToString("o"),
            parameters = workflow.Parameters,
            results = workflow.Results,
            history = ConvertHistory(workflow.History)
        };

        var resultJson = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(resultObj));
        return McpActionResult.CreateSuccess(resultJson, actionId);
    }

    /// <summary>
    /// Handles the stop workflow operation
    /// </summary>
    /// <param name="workflowId">ID of the workflow</param>
    /// <param name="actionId">The ID of the action</param>
    /// <returns>JSON response</returns>
    private async Task<McpActionResult> HandleStopWorkflowAsync(string workflowId, string actionId)
    {
        _logger.LogInformation($"Stopping workflow {workflowId}");

        // Validate parameters
        if (string.IsNullOrEmpty(workflowId))
        {
            return McpActionResult.CreateFailure("Workflow ID is required", actionId);
        }

        // Stop the workflow
        var success = await _workflowCoordinator.StopWorkflowAsync(workflowId);

        // Get the workflow status
        var workflow = await _workflowCoordinator.GetWorkflowStatusAsync(workflowId);

        // Record progress
        await _progressTracker.RecordProgressAsync(workflow);

        // Convert the workflow to a JSON-friendly format
        var resultObj = new
        {
            success = success,
            workflow_id = workflow.Id,
            workflow_type = workflow.Type,
            current_state = workflow.CurrentState,
            status = workflow.Status.ToString().ToLowerInvariant(),
            start_time = workflow.StartTime.ToString("o"),
            end_time = workflow.EndTime?.ToString("o")
        };

        var resultJson = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(resultObj));
        return McpActionResult.CreateSuccess(resultJson, actionId);
    }

    /// <summary>
    /// Handles the transition workflow operation
    /// </summary>
    /// <param name="parameters">Request parameters</param>
    /// <param name="actionId">The ID of the action</param>
    /// <returns>JSON response</returns>
    private async Task<McpActionResult> HandleTransitionWorkflowAsync(JsonElement parameters, string actionId)
    {
        _logger.LogInformation("Transitioning workflow");

        // Extract parameters
        var workflowId = parameters.TryGetProperty("workflow_id", out var workflowIdElement)
            ? workflowIdElement.GetString()
            : null;

        var nextState = parameters.TryGetProperty("next_state", out var nextStateElement)
            ? nextStateElement.GetString()
            : null;

        var resultObj = parameters.TryGetProperty("result", out var resultElement)
            ? resultElement
            : default;

        // Validate parameters
        if (string.IsNullOrEmpty(workflowId))
        {
            return McpActionResult.CreateFailure("Workflow ID is required", actionId);
        }

        if (string.IsNullOrEmpty(nextState))
        {
            return McpActionResult.CreateFailure("Next state is required", actionId);
        }

        // Convert the result to an object
        object result = null;
        if (resultObj.ValueKind != JsonValueKind.Undefined)
        {
            result = JsonSerializer.Deserialize<object>(resultObj.GetRawText());
        }

        // Transition the workflow
        var workflow = await _workflowCoordinator.TransitionWorkflowAsync(workflowId, nextState, result);

        // Record progress
        await _progressTracker.RecordProgressAsync(workflow);

        // Convert the workflow to a JSON-friendly format
        var responseObj = new
        {
            workflow_id = workflow.Id,
            workflow_type = workflow.Type,
            current_state = workflow.CurrentState,
            status = workflow.Status.ToString().ToLowerInvariant(),
            start_time = workflow.StartTime.ToString("o"),
            end_time = workflow.EndTime?.ToString("o")
        };

        var resultJson = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
        return McpActionResult.CreateSuccess(resultJson, actionId);
    }

    /// <summary>
    /// Handles the list workflows operation
    /// </summary>
    /// <param name="parameters">Request parameters</param>
    /// <param name="actionId">The ID of the action</param>
    /// <returns>JSON response</returns>
    private async Task<McpActionResult> HandleListWorkflowsAsync(JsonElement parameters, string actionId)
    {
        _logger.LogInformation("Listing workflows");

        // Extract parameters
        var workflowType = parameters.TryGetProperty("workflow_type", out var workflowTypeElement)
            ? workflowTypeElement.GetString()
            : null;

        var activeOnly = parameters.TryGetProperty("active_only", out var activeOnlyElement)
            ? activeOnlyElement.GetBoolean()
            : false;

        // Get the workflows
        List<WorkflowInstance> workflows;
        if (activeOnly)
        {
            workflows = await _workflowCoordinator.GetActiveWorkflowsAsync();
        }
        else if (!string.IsNullOrEmpty(workflowType))
        {
            workflows = await _workflowCoordinator.GetWorkflowsByTypeAsync(workflowType);
        }
        else
        {
            // Get all workflows (active and inactive)
            var activeWorkflows = await _workflowCoordinator.GetActiveWorkflowsAsync();
            var allWorkflowTypes = new[] { "SelfCoding" }; // Add more workflow types as needed
            workflows = new List<WorkflowInstance>(activeWorkflows);

            foreach (var type in allWorkflowTypes)
            {
                var typeWorkflows = await _workflowCoordinator.GetWorkflowsByTypeAsync(type);
                foreach (var workflow in typeWorkflows)
                {
                    if (!workflows.Exists(w => w.Id == workflow.Id))
                    {
                        workflows.Add(workflow);
                    }
                }
            }
        }

        // Convert the workflows to a JSON-friendly format
        var workflowList = new List<object>();
        foreach (var workflow in workflows)
        {
            workflowList.Add(new
            {
                workflow_id = workflow.Id,
                workflow_type = workflow.Type,
                current_state = workflow.CurrentState,
                status = workflow.Status.ToString().ToLowerInvariant(),
                start_time = workflow.StartTime.ToString("o"),
                end_time = workflow.EndTime?.ToString("o")
            });
        }

        var resultObj = new
        {
            workflows = workflowList,
            count = workflowList.Count
        };

        var resultJson = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(resultObj));
        return McpActionResult.CreateSuccess(resultJson, actionId);
    }

    /// <summary>
    /// Handles the generate report operation
    /// </summary>
    /// <param name="workflowId">ID of the workflow</param>
    /// <param name="actionId">The ID of the action</param>
    /// <returns>JSON response</returns>
    private async Task<McpActionResult> HandleGenerateReportAsync(string workflowId, string actionId)
    {
        _logger.LogInformation($"Generating report for workflow {workflowId}");

        // Validate parameters
        if (string.IsNullOrEmpty(workflowId))
        {
            return McpActionResult.CreateFailure("Workflow ID is required", actionId);
        }

        // Generate the report
        var report = await _progressTracker.GenerateProgressReportAsync(workflowId);

        // Calculate performance metrics
        var metrics = await _progressTracker.CalculatePerformanceMetricsAsync(workflowId);

        // Create the response
        var resultObj = new
        {
            workflow_id = workflowId,
            report,
            metrics
        };

        var resultJson = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(resultObj));
        return McpActionResult.CreateSuccess(resultJson, actionId);
    }

    /// <summary>
    /// Converts workflow state transitions to a JSON-friendly format
    /// </summary>
    /// <param name="history">List of workflow state transitions</param>
    /// <returns>List of JSON-friendly workflow state transitions</returns>
    private List<object> ConvertHistory(List<WorkflowStateTransition> history)
    {
        var result = new List<object>();
        foreach (var transition in history)
        {
            result.Add(new
            {
                from_state = transition.FromState,
                to_state = transition.ToState,
                timestamp = transition.Timestamp.ToString("o")
            });
        }
        return result;
    }
}