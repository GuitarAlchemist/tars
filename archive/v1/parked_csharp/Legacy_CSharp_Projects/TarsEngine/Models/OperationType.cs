namespace TarsEngine.Models;

/// <summary>
/// Represents the type of an operation
/// </summary>
public enum OperationType
{
    /// <summary>
    /// File creation operation
    /// </summary>
    FileCreation,

    /// <summary>
    /// File modification operation
    /// </summary>
    FileModification,

    /// <summary>
    /// File deletion operation
    /// </summary>
    FileDeletion,

    /// <summary>
    /// File backup operation
    /// </summary>
    FileBackup,

    /// <summary>
    /// File restore operation
    /// </summary>
    FileRestore,

    /// <summary>
    /// Command execution operation
    /// </summary>
    CommandExecution,

    /// <summary>
    /// Other operation type
    /// </summary>
    Other
}
