namespace TarsEngine.Models;

/// <summary>
/// Represents the type of an audit entry
/// </summary>
public enum AuditEntryType
{
    /// <summary>
    /// Context creation
    /// </summary>
    ContextCreation,

    /// <summary>
    /// Context completion
    /// </summary>
    ContextCompletion,

    /// <summary>
    /// Transaction begin
    /// </summary>
    TransactionBegin,

    /// <summary>
    /// Transaction commit
    /// </summary>
    TransactionCommit,

    /// <summary>
    /// Transaction rollback
    /// </summary>
    TransactionRollback,

    /// <summary>
    /// File creation
    /// </summary>
    FileCreation,

    /// <summary>
    /// File modification
    /// </summary>
    FileModification,

    /// <summary>
    /// File deletion
    /// </summary>
    FileDeletion,

    /// <summary>
    /// File backup
    /// </summary>
    FileBackup,

    /// <summary>
    /// File restore
    /// </summary>
    FileRestore,

    /// <summary>
    /// Command execution
    /// </summary>
    CommandExecution,

    /// <summary>
    /// Validation
    /// </summary>
    Validation,

    /// <summary>
    /// Test execution
    /// </summary>
    TestExecution,

    /// <summary>
    /// Error
    /// </summary>
    Error,

    /// <summary>
    /// Warning
    /// </summary>
    Warning,

    /// <summary>
    /// Information
    /// </summary>
    Information,

    /// <summary>
    /// Other
    /// </summary>
    Other
}
