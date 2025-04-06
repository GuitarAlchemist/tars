namespace TarsEngine.Models;

/// <summary>
/// Represents the status of an improvement
/// </summary>
public enum ImprovementStatus
{
    /// <summary>
    /// Improvement is pending
    /// </summary>
    Pending,

    /// <summary>
    /// Improvement is in progress
    /// </summary>
    InProgress,

    /// <summary>
    /// Improvement is completed
    /// </summary>
    Completed,

    /// <summary>
    /// Improvement is deferred
    /// </summary>
    Deferred,

    /// <summary>
    /// Improvement is rejected
    /// </summary>
    Rejected,

    /// <summary>
    /// Improvement is blocked
    /// </summary>
    Blocked,

    /// <summary>
    /// Improvement is scheduled
    /// </summary>
    Scheduled,

    /// <summary>
    /// Improvement is under review
    /// </summary>
    UnderReview,

    /// <summary>
    /// Improvement is approved
    /// </summary>
    Approved,

    /// <summary>
    /// Improvement is cancelled
    /// </summary>
    Cancelled,

    /// <summary>
    /// Improvement is merged
    /// </summary>
    Merged,

    /// <summary>
    /// Improvement is deployed
    /// </summary>
    Deployed
}
