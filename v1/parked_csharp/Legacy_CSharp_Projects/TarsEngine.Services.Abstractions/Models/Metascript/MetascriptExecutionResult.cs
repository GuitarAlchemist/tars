namespace TarsEngine.Services.Abstractions.Models.Metascript
{
    /// <summary>
    /// Represents the result of executing a Metascript.
    /// </summary>
    public class MetascriptExecutionResult
    {
        /// <summary>
        /// Gets or sets a value indicating whether the execution was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the output of the execution.
        /// </summary>
        public string Output { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the error message if the execution failed.
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// Gets or sets the execution time in milliseconds.
        /// </summary>
        public long ExecutionTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the status of the execution.
        /// </summary>
        public MetascriptExecutionStatus Status { get; set; } = MetascriptExecutionStatus.Unknown;

        /// <summary>
        /// Gets or sets the timestamp when the execution started.
        /// </summary>
        public DateTime StartTimestamp { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the timestamp when the execution completed.
        /// </summary>
        public DateTime EndTimestamp { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the logs generated during execution.
        /// </summary>
        public List<string> Logs { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the warnings generated during execution.
        /// </summary>
        public List<string> Warnings { get; set; } = new List<string>();
    }
}
