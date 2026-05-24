namespace TarsEngine.Models;

/// <summary>
/// Represents the type of an execution step
/// </summary>
public enum ExecutionStepType
{
    /// <summary>
    /// File modification step
    /// </summary>
    FileModification,

    /// <summary>
    /// File creation step
    /// </summary>
    FileCreation,

    /// <summary>
    /// File deletion step
    /// </summary>
    FileDeletion,

    /// <summary>
    /// File backup step
    /// </summary>
    FileBackup,

    /// <summary>
    /// File restore step
    /// </summary>
    FileRestore,

    /// <summary>
    /// Validation step
    /// </summary>
    Validation,

    /// <summary>
    /// Compilation step
    /// </summary>
    Compilation,

    /// <summary>
    /// Test execution step
    /// </summary>
    TestExecution,

    /// <summary>
    /// Command execution step
    /// </summary>
    CommandExecution,

    /// <summary>
    /// Notification step
    /// </summary>
    Notification,

    /// <summary>
    /// Approval step
    /// </summary>
    Approval,

    /// <summary>
    /// Rollback step
    /// </summary>
    Rollback,

    /// <summary>
    /// Other step type
    /// </summary>
    Other
}
