namespace TarsEngine.Models;

/// <summary>
/// Represents a context for rollback operations
/// </summary>
public class RollbackContext
{
    /// <summary>
    /// Gets or sets the context ID
    /// </summary>
    public string ContextId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the user ID
    /// </summary>
    public string UserId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the context description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the backup directory
    /// </summary>
    public string BackupDirectory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the context was created
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the context was completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }
}
