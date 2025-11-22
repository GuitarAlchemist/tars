namespace TarsEngine.Models;

/// <summary>
/// Represents the status of an execution step
/// </summary>
public enum ExecutionStepStatus
{
    /// <summary>
    /// The execution step is pending
    /// </summary>
    Pending,

    /// <summary>
    /// The execution step is in progress
    /// </summary>
    InProgress,

    /// <summary>
    /// The execution step is completed
    /// </summary>
    Completed,

    /// <summary>
    /// The execution step has failed
    /// </summary>
    Failed,

    /// <summary>
    /// The execution step has been skipped
    /// </summary>
    Skipped,

    /// <summary>
    /// The execution step has been cancelled
    /// </summary>
    Cancelled,

    /// <summary>
    /// The execution step has been rolled back
    /// </summary>
    RolledBack
}
