namespace TarsEngine.Models;

/// <summary>
/// Represents the status of an execution plan
/// </summary>
public enum ExecutionPlanStatus
{
    /// <summary>
    /// The execution plan has been created
    /// </summary>
    Created,

    /// <summary>
    /// The execution plan is being validated
    /// </summary>
    Validating,

    /// <summary>
    /// The execution plan is ready to execute
    /// </summary>
    Ready,

    /// <summary>
    /// The execution plan is in progress
    /// </summary>
    InProgress,

    /// <summary>
    /// The execution plan is paused
    /// </summary>
    Paused,

    /// <summary>
    /// The execution plan is completed
    /// </summary>
    Completed,

    /// <summary>
    /// The execution plan has failed
    /// </summary>
    Failed,

    /// <summary>
    /// The execution plan has been cancelled
    /// </summary>
    Cancelled,

    /// <summary>
    /// The execution plan has been rolled back
    /// </summary>
    RolledBack
}
