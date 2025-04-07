using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents a transaction for executing changes
/// </summary>
public class Transaction
{
    /// <summary>
    /// Gets or sets the transaction ID
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the context ID
    /// </summary>
    public string ContextId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the transaction name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the transaction description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the transaction status
    /// </summary>
    public TransactionStatus Status { get; set; } = TransactionStatus.Active;

    /// <summary>
    /// Gets or sets the timestamp when the transaction was started
    /// </summary>
    public DateTime StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the transaction was completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the operations in the transaction
    /// </summary>
    public List<Operation> Operations { get; set; } = new List<Operation>();
}
