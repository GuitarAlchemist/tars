namespace TarsEngine.Models;

/// <summary>
/// Represents an entry in the audit trail
/// </summary>
public class AuditEntry
{
    /// <summary>
    /// Gets or sets the entry ID
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the context ID
    /// </summary>
    public string ContextId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the entry type
    /// </summary>
    public AuditEntryType Type { get; set; }

    /// <summary>
    /// Gets or sets the target type
    /// </summary>
    public string TargetType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the target ID
    /// </summary>
    public string TargetId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the user ID
    /// </summary>
    public string UserId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the entry description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the entry details
    /// </summary>
    public string? Details { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the entry was recorded
    /// </summary>
    public DateTime Timestamp { get; set; }
}
