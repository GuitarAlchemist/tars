namespace TarsEngine.Metascripts
{
    /// <summary>
    /// Represents the result of executing a metascript.
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
        public string? Output { get; set; }

        /// <summary>
        /// Gets or sets the error message if the execution failed.
        /// </summary>
        public string? ErrorMessage { get; set; }
    }
}
