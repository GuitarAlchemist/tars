using TarsEngine.Models;

namespace TarsApp.Models
{
    /// <summary>
    /// Represents options for executing an improvement
    /// </summary>
    public class ExecutionOptions
    {
        /// <summary>
        /// Gets or sets a value indicating whether to execute in dry run mode (no actual changes)
        /// </summary>
        public bool DryRun { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to create backups of files before modifying them
        /// </summary>
        public bool Backup { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to validate changes after execution
        /// </summary>
        public bool Validate { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to automatically roll back changes if validation fails
        /// </summary>
        public bool AutoRollback { get; set; } = true;

        /// <summary>
        /// Gets or sets the execution environment
        /// </summary>
        public ExecutionEnvironment Environment { get; set; } = ExecutionEnvironment.Development;

        /// <summary>
        /// Gets or sets the output directory for generated files
        /// </summary>
        public string? OutputDirectory { get; set; }
    }
}
