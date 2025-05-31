using DistributedFileSync.Core.Models;

namespace DistributedFileSync.Core.Interfaces;

/// <summary>
/// Core synchronization engine interface
/// Designed by: Architect Agent (Alice)
/// Implemented by: Senior Developer Agent (Bob)
/// Performance optimized by: Performance Engineer Agent (Dave)
/// </summary>
public interface ISynchronizationEngine
{
    /// <summary>
    /// Start the synchronization engine
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the operation</returns>
    Task StartAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Stop the synchronization engine
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Task representing the operation</returns>
    Task StopAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Synchronize a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to synchronize</param>
    /// <param name="targetNodes">Target nodes to synchronize with</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Synchronization result</returns>
    Task<SyncResult> SynchronizeFileAsync(string filePath, IEnumerable<SyncNode> targetNodes, CancellationToken cancellationToken = default);

    /// <summary>
    /// Synchronize an entire directory
    /// </summary>
    /// <param name="directoryPath">Path to the directory to synchronize</param>
    /// <param name="targetNodes">Target nodes to synchronize with</param>
    /// <param name="recursive">Whether to synchronize subdirectories</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Synchronization result</returns>
    Task<SyncResult> SynchronizeDirectoryAsync(string directoryPath, IEnumerable<SyncNode> targetNodes, bool recursive = true, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get synchronization status for a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>File metadata with sync status</returns>
    Task<FileMetadata?> GetFileSyncStatusAsync(string filePath);

    /// <summary>
    /// Get all files currently being synchronized
    /// </summary>
    /// <returns>Collection of file metadata</returns>
    Task<IEnumerable<FileMetadata>> GetActiveSynchronizationsAsync();

    /// <summary>
    /// Resolve a file conflict
    /// </summary>
    /// <param name="fileId">File identifier</param>
    /// <param name="strategy">Resolution strategy</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Conflict resolution result</returns>
    Task<ConflictResolutionResult> ResolveConflictAsync(Guid fileId, ConflictResolutionStrategy strategy, CancellationToken cancellationToken = default);

    /// <summary>
    /// Event raised when a file synchronization starts
    /// </summary>
    event EventHandler<FileSyncEventArgs> SyncStarted;

    /// <summary>
    /// Event raised when a file synchronization completes
    /// </summary>
    event EventHandler<FileSyncEventArgs> SyncCompleted;

    /// <summary>
    /// Event raised when a file synchronization fails
    /// </summary>
    event EventHandler<FileSyncErrorEventArgs> SyncFailed;

    /// <summary>
    /// Event raised when a conflict is detected
    /// </summary>
    event EventHandler<ConflictDetectedEventArgs> ConflictDetected;
}

/// <summary>
/// Synchronization result
/// </summary>
public class SyncResult
{
    /// <summary>
    /// Whether the synchronization was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if synchronization failed
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Number of files synchronized
    /// </summary>
    public int FilesSynchronized { get; set; }

    /// <summary>
    /// Total bytes transferred
    /// </summary>
    public long BytesTransferred { get; set; }

    /// <summary>
    /// Time taken for synchronization
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Nodes that were synchronized with
    /// </summary>
    public List<SyncNode> SynchronizedNodes { get; set; } = new();

    /// <summary>
    /// Any conflicts that were detected
    /// </summary>
    public List<FileConflict> Conflicts { get; set; } = new();
}

/// <summary>
/// Conflict resolution result
/// </summary>
public class ConflictResolutionResult
{
    /// <summary>
    /// Whether the conflict was resolved
    /// </summary>
    public bool Resolved { get; set; }

    /// <summary>
    /// Resolution strategy that was used
    /// </summary>
    public ConflictResolutionStrategy Strategy { get; set; }

    /// <summary>
    /// Final file metadata after resolution
    /// </summary>
    public FileMetadata? ResolvedFile { get; set; }

    /// <summary>
    /// Error message if resolution failed
    /// </summary>
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// File conflict information
/// </summary>
public class FileConflict
{
    /// <summary>
    /// File identifier
    /// </summary>
    public Guid FileId { get; set; }

    /// <summary>
    /// File path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Local version of the file
    /// </summary>
    public FileMetadata LocalVersion { get; set; } = new();

    /// <summary>
    /// Remote version of the file
    /// </summary>
    public FileMetadata RemoteVersion { get; set; } = new();

    /// <summary>
    /// Type of conflict
    /// </summary>
    public ConflictType ConflictType { get; set; }

    /// <summary>
    /// When the conflict was detected
    /// </summary>
    public DateTime DetectedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Types of file conflicts
/// </summary>
public enum ConflictType
{
    ModificationConflict,
    DeletionConflict,
    CreationConflict,
    PermissionConflict
}

/// <summary>
/// File synchronization event arguments
/// </summary>
public class FileSyncEventArgs : EventArgs
{
    public FileMetadata FileMetadata { get; set; } = new();
    public SyncNode TargetNode { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// File synchronization error event arguments
/// </summary>
public class FileSyncErrorEventArgs : FileSyncEventArgs
{
    public string ErrorMessage { get; set; } = string.Empty;
    public Exception? Exception { get; set; }
}

/// <summary>
/// Conflict detected event arguments
/// </summary>
public class ConflictDetectedEventArgs : EventArgs
{
    public FileConflict Conflict { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}
