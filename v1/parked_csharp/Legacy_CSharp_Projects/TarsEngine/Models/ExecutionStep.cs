namespace TarsEngine.Models;

/// <summary>
/// Represents a step in an execution plan
/// </summary>
public class ExecutionStep
{
    /// <summary>
    /// Gets or sets the ID of the execution step
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the execution step
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the execution step
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of the execution step
    /// </summary>
    public ExecutionStepType Type { get; set; } = ExecutionStepType.Other;

    /// <summary>
    /// Gets or sets the order of the execution step
    /// </summary>
    public int Order { get; set; }

    /// <summary>
    /// Gets or sets the dependencies of the execution step
    /// </summary>
    public List<string> Dependencies { get; set; } = new();

    /// <summary>
    /// Gets or sets the status of the execution step
    /// </summary>
    public ExecutionStepStatus Status { get; set; } = ExecutionStepStatus.Pending;

    /// <summary>
    /// Gets or sets the result of the execution step
    /// </summary>
    public ExecutionStepResult? Result { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution step was started
    /// </summary>
    public DateTime? StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution step was completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the duration of the execution step in milliseconds
    /// </summary>
    public long? DurationMs { get; set; }

    /// <summary>
    /// Gets or sets the action to execute
    /// </summary>
    public string Action { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the parameters for the action
    /// </summary>
    public Dictionary<string, string> Parameters { get; set; } = new();

    /// <summary>
    /// Gets or sets the validation rules for the execution step
    /// </summary>
    public List<ValidationRule> ValidationRules { get; set; } = new();

    /// <summary>
    /// Gets or sets the rollback action
    /// </summary>
    public string RollbackAction { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the parameters for the rollback action
    /// </summary>
    public Dictionary<string, string> RollbackParameters { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata about the execution step
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets whether the execution step is completed
    /// </summary>
    public bool IsCompleted => Status == ExecutionStepStatus.Completed || Status == ExecutionStepStatus.Failed || Status == ExecutionStepStatus.Skipped;

    /// <summary>
    /// Gets whether the execution step is successful
    /// </summary>
    public bool IsSuccessful => Status == ExecutionStepStatus.Completed && (Result?.IsSuccessful ?? false);

    /// <summary>
    /// Gets whether the execution step is ready to execute
    /// </summary>
    /// <param name="completedStepIds">The IDs of completed steps</param>
    /// <returns>True if the execution step is ready to execute, false otherwise</returns>
    public bool IsReadyToExecute(HashSet<string> completedStepIds)
    {
        if (Status != ExecutionStepStatus.Pending)
        {
            return false;
        }

        foreach (var dependencyId in Dependencies)
        {
            if (!completedStepIds.Contains(dependencyId))
            {
                return false;
            }
        }

        return true;
    }
}
