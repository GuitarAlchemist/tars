namespace TarsCli.Services.Workflow;

/// <summary>
/// Interface for workflow coordinators
/// </summary>
public interface IWorkflowCoordinator
{
    /// <summary>
    /// Starts a workflow
    /// </summary>
    /// <param name="workflowType">Type of workflow to start</param>
    /// <param name="parameters">Workflow parameters</param>
    /// <returns>Workflow instance</returns>
    Task<WorkflowInstance> StartWorkflowAsync(string workflowType, Dictionary<string, object> parameters);

    /// <summary>
    /// Gets the status of a workflow
    /// </summary>
    /// <param name="workflowId">ID of the workflow</param>
    /// <returns>Workflow instance</returns>
    Task<WorkflowInstance> GetWorkflowStatusAsync(string workflowId);

    /// <summary>
    /// Stops a workflow
    /// </summary>
    /// <param name="workflowId">ID of the workflow</param>
    /// <returns>True if the workflow was stopped successfully, false otherwise</returns>
    Task<bool> StopWorkflowAsync(string workflowId);

    /// <summary>
    /// Gets all active workflows
    /// </summary>
    /// <returns>List of active workflow instances</returns>
    Task<List<WorkflowInstance>> GetActiveWorkflowsAsync();

    /// <summary>
    /// Gets all workflows of a specific type
    /// </summary>
    /// <param name="workflowType">Type of workflow</param>
    /// <returns>List of workflow instances</returns>
    Task<List<WorkflowInstance>> GetWorkflowsByTypeAsync(string workflowType);

    /// <summary>
    /// Transitions a workflow to the next state
    /// </summary>
    /// <param name="workflowId">ID of the workflow</param>
    /// <param name="nextState">Next state to transition to</param>
    /// <param name="result">Result of the current state</param>
    /// <returns>Updated workflow instance</returns>
    Task<WorkflowInstance> TransitionWorkflowAsync(string workflowId, string nextState, object result);
}

/// <summary>
/// Represents a workflow instance
/// </summary>
public class WorkflowInstance
{
    /// <summary>
    /// ID of the workflow
    /// </summary>
    public string Id { get; set; }

    /// <summary>
    /// Type of the workflow
    /// </summary>
    public string Type { get; set; }

    /// <summary>
    /// Current state of the workflow
    /// </summary>
    public string CurrentState { get; set; }

    /// <summary>
    /// Status of the workflow
    /// </summary>
    public WorkflowStatus Status { get; set; }

    /// <summary>
    /// Parameters of the workflow
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; } = new();

    /// <summary>
    /// Results of the workflow
    /// </summary>
    public Dictionary<string, object> Results { get; set; } = new();

    /// <summary>
    /// History of the workflow
    /// </summary>
    public List<WorkflowStateTransition> History { get; set; } = new();

    /// <summary>
    /// Error message if the workflow failed
    /// </summary>
    public string ErrorMessage { get; set; }

    /// <summary>
    /// Start time of the workflow
    /// </summary>
    public System.DateTime StartTime { get; set; }

    /// <summary>
    /// End time of the workflow
    /// </summary>
    public System.DateTime? EndTime { get; set; }
}

/// <summary>
/// Represents a workflow state transition
/// </summary>
public class WorkflowStateTransition
{
    /// <summary>
    /// Previous state
    /// </summary>
    public string FromState { get; set; }

    /// <summary>
    /// Next state
    /// </summary>
    public string ToState { get; set; }

    /// <summary>
    /// Timestamp of the transition
    /// </summary>
    public System.DateTime Timestamp { get; set; }

    /// <summary>
    /// Result of the state
    /// </summary>
    public object Result { get; set; }
}

/// <summary>
/// Status of a workflow
/// </summary>
public enum WorkflowStatus
{
    /// <summary>
    /// Workflow is created but not started
    /// </summary>
    Created,

    /// <summary>
    /// Workflow is running
    /// </summary>
    Running,

    /// <summary>
    /// Workflow is paused
    /// </summary>
    Paused,

    /// <summary>
    /// Workflow is completed successfully
    /// </summary>
    Completed,

    /// <summary>
    /// Workflow failed
    /// </summary>
    Failed,

    /// <summary>
    /// Workflow was cancelled
    /// </summary>
    Cancelled
}