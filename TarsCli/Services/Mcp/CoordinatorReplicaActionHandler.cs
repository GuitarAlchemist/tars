using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.Workflow;

namespace TarsCli.Services.Mcp
{
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
        public string ActionName => "coordinate_workflow";

        /// <inheritdoc/>
        public async Task<JsonElement> HandleActionAsync(JsonElement request)
        {
            _logger.LogInformation("Handling coordinate_workflow action");

            try
            {
                // Extract parameters from the request
                var operation = request.TryGetProperty("operation", out var operationElement)
                    ? operationElement.GetString()
                    : "status";

                var workflowId = request.TryGetProperty("workflow_id", out var workflowIdElement)
                    ? workflowIdElement.GetString()
                    : null;

                // Handle different operations
                switch (operation)
                {
                    case "start":
                        return await HandleStartWorkflowAsync(request);

                    case "status":
                        return await HandleGetWorkflowStatusAsync(workflowId);

                    case "stop":
                        return await HandleStopWorkflowAsync(workflowId);

                    case "transition":
                        return await HandleTransitionWorkflowAsync(request);

                    case "list":
                        return await HandleListWorkflowsAsync(request);

                    case "report":
                        return await HandleGenerateReportAsync(workflowId);

                    default:
                        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                        {
                            success = false,
                            error = $"Unknown operation: {operation}"
                        }));
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling coordinate_workflow action");
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = ex.Message
                }));
            }
        }

        /// <summary>
        /// Handles the start workflow operation
        /// </summary>
        /// <param name="request">Request parameters</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleStartWorkflowAsync(JsonElement request)
        {
            _logger.LogInformation("Starting workflow");

            // Extract parameters
            var workflowType = request.TryGetProperty("workflow_type", out var workflowTypeElement)
                ? workflowTypeElement.GetString()
                : "SelfCoding";

            var parameters = new Dictionary<string, object>();
            if (request.TryGetProperty("parameters", out var parametersElement) && parametersElement.ValueKind == JsonValueKind.Object)
            {
                foreach (var parameter in parametersElement.EnumerateObject())
                {
                    switch (parameter.Value.ValueKind)
                    {
                        case JsonValueKind.String:
                            parameters[parameter.Name] = parameter.Value.GetString();
                            break;
                        case JsonValueKind.Number:
                            parameters[parameter.Name] = parameter.Value.GetDouble();
                            break;
                        case JsonValueKind.True:
                        case JsonValueKind.False:
                            parameters[parameter.Name] = parameter.Value.GetBoolean();
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
                            parameters[parameter.Name] = list;
                            break;
                    }
                }
            }

            // Start the workflow
            var workflow = await _workflowCoordinator.StartWorkflowAsync(workflowType, parameters);

            // Record progress
            await _progressTracker.RecordProgressAsync(workflow);

            // Convert the workflow to a JSON-friendly format
            var result = new
            {
                success = true,
                workflow_id = workflow.Id,
                workflow_type = workflow.Type,
                current_state = workflow.CurrentState,
                status = workflow.Status.ToString().ToLowerInvariant(),
                start_time = workflow.StartTime.ToString("o")
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
        }

        /// <summary>
        /// Handles the get workflow status operation
        /// </summary>
        /// <param name="workflowId">ID of the workflow</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleGetWorkflowStatusAsync(string workflowId)
        {
            _logger.LogInformation($"Getting status of workflow {workflowId}");

            // Validate parameters
            if (string.IsNullOrEmpty(workflowId))
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Workflow ID is required"
                }));
            }

            // Get the workflow status
            var workflow = await _workflowCoordinator.GetWorkflowStatusAsync(workflowId);

            // Convert the workflow to a JSON-friendly format
            var result = new
            {
                success = true,
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

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
        }

        /// <summary>
        /// Handles the stop workflow operation
        /// </summary>
        /// <param name="workflowId">ID of the workflow</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleStopWorkflowAsync(string workflowId)
        {
            _logger.LogInformation($"Stopping workflow {workflowId}");

            // Validate parameters
            if (string.IsNullOrEmpty(workflowId))
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Workflow ID is required"
                }));
            }

            // Stop the workflow
            var success = await _workflowCoordinator.StopWorkflowAsync(workflowId);

            // Get the workflow status
            var workflow = await _workflowCoordinator.GetWorkflowStatusAsync(workflowId);

            // Record progress
            await _progressTracker.RecordProgressAsync(workflow);

            // Convert the workflow to a JSON-friendly format
            var result = new
            {
                success = success,
                workflow_id = workflow.Id,
                workflow_type = workflow.Type,
                current_state = workflow.CurrentState,
                status = workflow.Status.ToString().ToLowerInvariant(),
                start_time = workflow.StartTime.ToString("o"),
                end_time = workflow.EndTime?.ToString("o")
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
        }

        /// <summary>
        /// Handles the transition workflow operation
        /// </summary>
        /// <param name="request">Request parameters</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleTransitionWorkflowAsync(JsonElement request)
        {
            _logger.LogInformation("Transitioning workflow");

            // Extract parameters
            var workflowId = request.TryGetProperty("workflow_id", out var workflowIdElement)
                ? workflowIdElement.GetString()
                : null;

            var nextState = request.TryGetProperty("next_state", out var nextStateElement)
                ? nextStateElement.GetString()
                : null;

            var resultObj = request.TryGetProperty("result", out var resultElement)
                ? resultElement
                : default;

            // Validate parameters
            if (string.IsNullOrEmpty(workflowId))
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Workflow ID is required"
                }));
            }

            if (string.IsNullOrEmpty(nextState))
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Next state is required"
                }));
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
            var response = new
            {
                success = true,
                workflow_id = workflow.Id,
                workflow_type = workflow.Type,
                current_state = workflow.CurrentState,
                status = workflow.Status.ToString().ToLowerInvariant(),
                start_time = workflow.StartTime.ToString("o"),
                end_time = workflow.EndTime?.ToString("o")
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(response));
        }

        /// <summary>
        /// Handles the list workflows operation
        /// </summary>
        /// <param name="request">Request parameters</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleListWorkflowsAsync(JsonElement request)
        {
            _logger.LogInformation("Listing workflows");

            // Extract parameters
            var workflowType = request.TryGetProperty("workflow_type", out var workflowTypeElement)
                ? workflowTypeElement.GetString()
                : null;

            var activeOnly = request.TryGetProperty("active_only", out var activeOnlyElement)
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

            var result = new
            {
                success = true,
                workflows = workflowList,
                count = workflowList.Count
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
        }

        /// <summary>
        /// Handles the generate report operation
        /// </summary>
        /// <param name="workflowId">ID of the workflow</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleGenerateReportAsync(string workflowId)
        {
            _logger.LogInformation($"Generating report for workflow {workflowId}");

            // Validate parameters
            if (string.IsNullOrEmpty(workflowId))
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Workflow ID is required"
                }));
            }

            // Generate the report
            var report = await _progressTracker.GenerateProgressReportAsync(workflowId);

            // Calculate performance metrics
            var metrics = await _progressTracker.CalculatePerformanceMetricsAsync(workflowId);

            // Create the response
            var result = new
            {
                success = true,
                workflow_id = workflowId,
                report,
                metrics
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
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
}
