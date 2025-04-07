namespace TarsEngine.Models;

/// <summary>
/// Represents the status of a transaction
/// </summary>
public enum TransactionStatus
{
    /// <summary>
    /// The transaction is active
    /// </summary>
    Active,

    /// <summary>
    /// The transaction is committed
    /// </summary>
    Committed,

    /// <summary>
    /// The transaction is rolled back
    /// </summary>
    RolledBack,

    /// <summary>
    /// The transaction is aborted
    /// </summary>
    Aborted
}
