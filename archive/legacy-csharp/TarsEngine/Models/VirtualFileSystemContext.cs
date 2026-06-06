namespace TarsEngine.Models;

/// <summary>
/// Represents a virtual file system context
/// </summary>
public class VirtualFileSystemContext
{
    /// <summary>
    /// Gets or sets the context ID
    /// </summary>
    public string ContextId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the working directory
    /// </summary>
    public string WorkingDirectory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the backup directory
    /// </summary>
    public string BackupDirectory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the context is in dry run mode
    /// </summary>
    public bool IsDryRun { get; set; } = true;

    /// <summary>
    /// Gets or sets the timestamp when the context was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the context was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Gets or sets the virtual files
    /// </summary>
    public Dictionary<string, VirtualFile> VirtualFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of modified files
    /// </summary>
    public HashSet<string> ModifiedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of created files
    /// </summary>
    public HashSet<string> CreatedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of deleted files
    /// </summary>
    public HashSet<string> DeletedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of backed up files
    /// </summary>
    public HashSet<string> BackedUpFiles { get; set; } = new();
}
