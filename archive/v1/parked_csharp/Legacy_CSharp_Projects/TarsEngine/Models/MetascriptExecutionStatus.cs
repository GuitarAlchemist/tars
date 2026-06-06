namespace TarsEngine.Models;

/// <summary>
/// Represents the execution status of a metascript
/// </summary>
public enum MetascriptExecutionStatus
{
    /// <summary>
    /// Metascript has not been executed
    /// </summary>
    NotExecuted,

    /// <summary>
    /// Metascript is currently executing
    /// </summary>
    Executing,

    /// <summary>
    /// Metascript execution completed successfully
    /// </summary>
    Succeeded,

    /// <summary>
    /// Metascript execution failed
    /// </summary>
    Failed,

    /// <summary>
    /// Metascript execution timed out
    /// </summary>
    TimedOut,

    /// <summary>
    /// Metascript execution was cancelled
    /// </summary>
    Cancelled,

    /// <summary>
    /// Metascript execution is pending
    /// </summary>
    Pending,

    /// <summary>
    /// Metascript execution is queued
    /// </summary>
    Queued,

    /// <summary>
    /// Metascript execution is paused
    /// </summary>
    Paused,

    /// <summary>
    /// Metascript execution is scheduled
    /// </summary>
    Scheduled,

    /// <summary>
    /// Metascript execution status is unknown
    /// </summary>
    Unknown
}
