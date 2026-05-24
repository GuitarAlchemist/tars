namespace TarsEngine.Services.Abstractions.Models.Metascript
{
    /// <summary>
    /// Represents the status of a Metascript execution.
    /// </summary>
    public enum MetascriptExecutionStatus
    {
        /// <summary>
        /// The execution status is unknown.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// The execution is pending.
        /// </summary>
        Pending = 1,

        /// <summary>
        /// The execution is in progress.
        /// </summary>
        InProgress = 2,

        /// <summary>
        /// The execution completed successfully.
        /// </summary>
        Completed = 3,

        /// <summary>
        /// The execution failed.
        /// </summary>
        Failed = 4,

        /// <summary>
        /// The execution timed out.
        /// </summary>
        TimedOut = 5,

        /// <summary>
        /// The execution was cancelled.
        /// </summary>
        Cancelled = 6
    }
}
