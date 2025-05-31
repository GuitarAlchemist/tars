using System.Text.Json.Serialization;

namespace DistributedFileSync.Core.Models;

/// <summary>
/// Represents metadata for a synchronized file
/// Designed by: Architect Agent (Alice)
/// Implemented by: Senior Developer Agent (Bob)
/// </summary>
public class FileMetadata
{
    /// <summary>
    /// Unique identifier for the file
    /// </summary>
    public Guid Id { get; set; } = Guid.NewGuid();

    /// <summary>
    /// Full path to the file
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// File name without path
    /// </summary>
    public string FileName { get; set; } = string.Empty;

    /// <summary>
    /// File size in bytes
    /// </summary>
    public long FileSize { get; set; }

    /// <summary>
    /// SHA-256 hash of the file content
    /// </summary>
    public string ContentHash { get; set; } = string.Empty;

    /// <summary>
    /// Last modification time of the file
    /// </summary>
    public DateTime LastModified { get; set; }

    /// <summary>
    /// Creation time of the file
    /// </summary>
    public DateTime CreatedAt { get; set; }

    /// <summary>
    /// Version number for conflict resolution
    /// </summary>
    public long Version { get; set; } = 1;

    /// <summary>
    /// Node ID that last modified the file
    /// </summary>
    public string LastModifiedBy { get; set; } = string.Empty;

    /// <summary>
    /// Indicates if the file is currently being synchronized
    /// </summary>
    public bool IsSyncing { get; set; }

    /// <summary>
    /// Indicates if there's a conflict with this file
    /// </summary>
    public bool HasConflict { get; set; }

    /// <summary>
    /// Conflict resolution strategy
    /// </summary>
    public ConflictResolutionStrategy ConflictStrategy { get; set; } = ConflictResolutionStrategy.ThreeWayMerge;

    /// <summary>
    /// File synchronization status
    /// </summary>
    public SyncStatus Status { get; set; } = SyncStatus.Pending;

    /// <summary>
    /// Additional metadata as key-value pairs
    /// </summary>
    public Dictionary<string, string> AdditionalMetadata { get; set; } = new();

    /// <summary>
    /// Timestamp when this metadata was last updated
    /// </summary>
    public DateTime MetadataUpdated { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// File synchronization status
/// </summary>
public enum SyncStatus
{
    Pending,
    InProgress,
    Completed,
    Failed,
    Conflict,
    Deleted
}

/// <summary>
/// Conflict resolution strategies
/// Researched by: Researcher Agent (Carol)
/// </summary>
public enum ConflictResolutionStrategy
{
    LastWriteWins,
    ThreeWayMerge,
    ManualResolution,
    KeepBoth
}
