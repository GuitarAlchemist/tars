namespace TarsApp.Models
{
    /// <summary>
    /// Represents options for rolling back changes
    /// </summary>
    public class RollbackOptions
    {
        /// <summary>
        /// Gets or sets a value indicating whether to roll back all changes
        /// </summary>
        public bool All { get; set; }

        /// <summary>
        /// Gets or sets the transaction ID to roll back
        /// </summary>
        public string? TransactionId { get; set; }

        /// <summary>
        /// Gets or sets the file path to roll back
        /// </summary>
        public string? FilePath { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to force the rollback
        /// </summary>
        public bool Force { get; set; }
    }
}
