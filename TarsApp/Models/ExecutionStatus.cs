namespace TarsApp.Models
{
    /// <summary>
    /// Represents the status of an execution
    /// </summary>
    public enum ExecutionStatus
    {
        /// <summary>
        /// Execution is in progress
        /// </summary>
        InProgress,
        
        /// <summary>
        /// Execution has completed successfully
        /// </summary>
        Completed,
        
        /// <summary>
        /// Execution has failed
        /// </summary>
        Failed,
        
        /// <summary>
        /// Execution is awaiting review
        /// </summary>
        AwaitingReview
    }
}
