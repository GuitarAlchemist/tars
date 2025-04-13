namespace TarsCli.Services.Workflow;

/// <summary>
/// Implementation of the workflow coordinator
/// </summary>
public class WorkflowCoordinator : IWorkflowCoordinator
{
    private readonly ILogger<WorkflowCoordinator> _logger;
    private readonly Dictionary<string, WorkflowInstance> _workflows = new();
    private readonly Dictionary<string, IWorkflowDefinition> _workflowDefinitions = new();

    /// <summary>
    /// Initializes a new instance of the WorkflowCoordinator class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="workflowDefinitions">Collection of workflow definitions</param>
    public WorkflowCoordinator(ILogger<WorkflowCoordinator> logger, IEnumerable<IWorkflowDefinition> workflowDefinitions)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
        if (workflowDefinitions != null)
        {
            foreach (var definition in workflowDefinitions)
            {
                _workflowDefinitions[definition.Type] = definition;
            }
        }
    }

    /// <inheritdoc/>
    public async Task<WorkflowInstance> StartWorkflowAsync(string workflowType, Dictionary<string, object> parameters)
    {
        _logger.LogInformation($"Starting workflow of type {workflowType}");

        try
        {
            // Check if the workflow type is supported
            if (!_workflowDefinitions.TryGetValue(workflowType, out var definition))
            {
                _logger.LogError($"Workflow type {workflowType} is not supported");
                throw new ArgumentException($"Workflow type {workflowType} is not supported");
            }

            // Validate parameters
            var validationResult = definition.ValidateParameters(parameters);
            if (!validationResult.IsValid)
            {
                _logger.LogError($"Invalid parameters for workflow type {workflowType}: {validationResult.ErrorMessage}");
                throw new ArgumentException($"Invalid parameters for workflow type {workflowType}: {validationResult.ErrorMessage}");
            }

            // Create a new workflow instance
            var workflowId = Guid.NewGuid().ToString();
            var initialState = definition.GetInitialState();
            var workflow = new WorkflowInstance
            {
                Id = workflowId,
                Type = workflowType,
                CurrentState = initialState,
                Status = WorkflowStatus.Created,
                Parameters = parameters,
                StartTime = DateTime.UtcNow
            };

            // Add the workflow to the collection
            _workflows[workflowId] = workflow;

            // Start the workflow
            workflow.Status = WorkflowStatus.Running;
            _logger.LogInformation($"Workflow {workflowId} of type {workflowType} started");

            // Execute the initial state
            await ExecuteStateAsync(workflow, initialState);

            return workflow;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error starting workflow of type {workflowType}");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<WorkflowInstance> GetWorkflowStatusAsync(string workflowId)
    {
        _logger.LogInformation($"Getting status of workflow {workflowId}");

        try
        {
            // Check if the workflow exists
            if (!_workflows.TryGetValue(workflowId, out var workflow))
            {
                _logger.LogError($"Workflow {workflowId} not found");
                throw new ArgumentException($"Workflow {workflowId} not found");
            }

            return await Task.FromResult(workflow);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting status of workflow {workflowId}");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> StopWorkflowAsync(string workflowId)
    {
        _logger.LogInformation($"Stopping workflow {workflowId}");

        try
        {
            // Check if the workflow exists
            if (!_workflows.TryGetValue(workflowId, out var workflow))
            {
                _logger.LogError($"Workflow {workflowId} not found");
                throw new ArgumentException($"Workflow {workflowId} not found");
            }

            // Check if the workflow is already completed or cancelled
            if (workflow.Status == WorkflowStatus.Completed || workflow.Status == WorkflowStatus.Failed || workflow.Status == WorkflowStatus.Cancelled)
            {
                _logger.LogWarning($"Workflow {workflowId} is already in state {workflow.Status}");
                return false;
            }

            // Stop the workflow
            workflow.Status = WorkflowStatus.Cancelled;
            workflow.EndTime = DateTime.UtcNow;
            _logger.LogInformation($"Workflow {workflowId} stopped");

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error stopping workflow {workflowId}");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<WorkflowInstance>> GetActiveWorkflowsAsync()
    {
        _logger.LogInformation("Getting active workflows");

        try
        {
            var activeWorkflows = _workflows.Values
                .Where(w => w.Status == WorkflowStatus.Running || w.Status == WorkflowStatus.Paused)
                .ToList();

            return await Task.FromResult(activeWorkflows);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting active workflows");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<WorkflowInstance>> GetWorkflowsByTypeAsync(string workflowType)
    {
        _logger.LogInformation($"Getting workflows of type {workflowType}");

        try
        {
            var workflows = _workflows.Values
                .Where(w => w.Type == workflowType)
                .ToList();

            return await Task.FromResult(workflows);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting workflows of type {workflowType}");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<WorkflowInstance> TransitionWorkflowAsync(string workflowId, string nextState, object result)
    {
        _logger.LogInformation($"Transitioning workflow {workflowId} to state {nextState}");

        try
        {
            // Check if the workflow exists
            if (!_workflows.TryGetValue(workflowId, out var workflow))
            {
                _logger.LogError($"Workflow {workflowId} not found");
                throw new ArgumentException($"Workflow {workflowId} not found");
            }

            // Check if the workflow is running
            if (workflow.Status != WorkflowStatus.Running)
            {
                _logger.LogError($"Workflow {workflowId} is not running");
                throw new InvalidOperationException($"Workflow {workflowId} is not running");
            }

            // Check if the workflow definition exists
            if (!_workflowDefinitions.TryGetValue(workflow.Type, out var definition))
            {
                _logger.LogError($"Workflow definition for type {workflow.Type} not found");
                throw new InvalidOperationException($"Workflow definition for type {workflow.Type} not found");
            }

            // Check if the transition is valid
            if (!definition.IsValidTransition(workflow.CurrentState, nextState))
            {
                _logger.LogError($"Invalid transition from {workflow.CurrentState} to {nextState} for workflow {workflowId}");
                throw new InvalidOperationException($"Invalid transition from {workflow.CurrentState} to {nextState} for workflow {workflowId}");
            }

            // Record the transition
            var transition = new WorkflowStateTransition
            {
                FromState = workflow.CurrentState,
                ToState = nextState,
                Timestamp = DateTime.UtcNow,
                Result = result
            };
            workflow.History.Add(transition);

            // Update the current state
            var previousState = workflow.CurrentState;
            workflow.CurrentState = nextState;

            // Store the result
            if (result != null)
            {
                workflow.Results[previousState] = result;
            }

            _logger.LogInformation($"Workflow {workflowId} transitioned from {previousState} to {nextState}");

            // Execute the new state
            await ExecuteStateAsync(workflow, nextState);

            return workflow;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error transitioning workflow {workflowId} to state {nextState}");
            throw;
        }
    }

    /// <summary>
    /// Executes a workflow state
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <param name="state">State to execute</param>
    /// <returns>Task representing the asynchronous operation</returns>
    private async Task ExecuteStateAsync(WorkflowInstance workflow, string state)
    {
        _logger.LogInformation($"Executing state {state} for workflow {workflow.Id}");

        try
        {
            // Check if the workflow definition exists
            if (!_workflowDefinitions.TryGetValue(workflow.Type, out var definition))
            {
                _logger.LogError($"Workflow definition for type {workflow.Type} not found");
                throw new InvalidOperationException($"Workflow definition for type {workflow.Type} not found");
            }

            // Execute the state
            var result = await definition.ExecuteStateAsync(workflow, state);

            // Check if this is a final state
            if (definition.IsFinalState(state))
            {
                workflow.Status = WorkflowStatus.Completed;
                workflow.EndTime = DateTime.UtcNow;
                _logger.LogInformation($"Workflow {workflow.Id} completed");
            }
            else
            {
                // Get the next state
                var nextState = definition.GetNextState(state, result);
                if (nextState != null)
                {
                    // Transition to the next state
                    await TransitionWorkflowAsync(workflow.Id, nextState, result);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing state {state} for workflow {workflow.Id}");
            workflow.Status = WorkflowStatus.Failed;
            workflow.ErrorMessage = ex.Message;
            workflow.EndTime = DateTime.UtcNow;
            throw;
        }
    }
}