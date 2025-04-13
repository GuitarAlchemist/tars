using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Manages transactions for executing changes
/// </summary>
public class TransactionManager
{
    private readonly ILogger<TransactionManager> _logger;
    private readonly FileBackupService _fileBackupService;
    private readonly Dictionary<string, Transaction> _transactions = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="TransactionManager"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="fileBackupService">The file backup service</param>
    public TransactionManager(ILogger<TransactionManager> logger, FileBackupService fileBackupService)
    {
        _logger = logger;
        _fileBackupService = fileBackupService;
    }

    /// <summary>
    /// Begins a transaction
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="name">The transaction name</param>
    /// <param name="description">The transaction description</param>
    /// <returns>The transaction ID</returns>
    public string BeginTransaction(string contextId, string name, string description)
    {
        try
        {
            _logger.LogInformation("Beginning transaction: {Name}", name);

            // Create transaction
            var transaction = new Transaction
            {
                Id = Guid.NewGuid().ToString(),
                ContextId = contextId,
                Name = name,
                Description = description,
                Status = TransactionStatus.Active,
                StartedAt = DateTime.UtcNow
            };

            // Store transaction
            _transactions[transaction.Id] = transaction;

            _logger.LogInformation("Transaction begun: {TransactionId}", transaction.Id);
            return transaction.Id;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error beginning transaction: {Name}", name);
            throw;
        }
    }

    /// <summary>
    /// Commits a transaction
    /// </summary>
    /// <param name="transactionId">The transaction ID</param>
    /// <returns>True if the transaction was committed successfully, false otherwise</returns>
    public bool CommitTransaction(string transactionId)
    {
        try
        {
            _logger.LogInformation("Committing transaction: {TransactionId}", transactionId);

            // Check if transaction exists
            if (!_transactions.TryGetValue(transactionId, out var transaction))
            {
                _logger.LogWarning("Transaction not found: {TransactionId}", transactionId);
                return false;
            }

            // Check if transaction is active
            if (transaction.Status != TransactionStatus.Active)
            {
                _logger.LogWarning("Transaction is not active: {TransactionId}, Status: {Status}", transactionId, transaction.Status);
                return false;
            }

            // Update transaction
            transaction.Status = TransactionStatus.Committed;
            transaction.CompletedAt = DateTime.UtcNow;

            _logger.LogInformation("Transaction committed: {TransactionId}", transactionId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error committing transaction: {TransactionId}", transactionId);
            return false;
        }
    }

    /// <summary>
    /// Rolls back a transaction
    /// </summary>
    /// <param name="transactionId">The transaction ID</param>
    /// <returns>True if the transaction was rolled back successfully, false otherwise</returns>
    public async Task<bool> RollbackTransactionAsync(string transactionId)
    {
        try
        {
            _logger.LogInformation("Rolling back transaction: {TransactionId}", transactionId);

            // Check if transaction exists
            if (!_transactions.TryGetValue(transactionId, out var transaction))
            {
                _logger.LogWarning("Transaction not found: {TransactionId}", transactionId);
                return false;
            }

            // Check if transaction is active or committed
            if (transaction.Status != TransactionStatus.Active && transaction.Status != TransactionStatus.Committed)
            {
                _logger.LogWarning("Transaction cannot be rolled back: {TransactionId}, Status: {Status}", transactionId, transaction.Status);
                return false;
            }

            // Restore files
            var success = true;
            foreach (var operation in transaction.Operations.OrderByDescending(o => o.Timestamp))
            {
                if (operation.Type == OperationType.FileModification ||
                    operation.Type == OperationType.FileDeletion)
                {
                    var filePath = operation.Target;
                    if (_fileBackupService.IsFileBackedUp(transaction.ContextId, filePath))
                    {
                        var result = await _fileBackupService.RestoreFileAsync(transaction.ContextId, filePath);
                        if (!result)
                        {
                            _logger.LogWarning("Failed to restore file: {FilePath}", filePath);
                            success = false;
                        }
                    }
                }
                else if (operation.Type == OperationType.FileCreation)
                {
                    var filePath = operation.Target;
                    if (System.IO.File.Exists(filePath))
                    {
                        try
                        {
                            System.IO.File.Delete(filePath);
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Failed to delete created file: {FilePath}", filePath);
                            success = false;
                        }
                    }
                }
            }

            // Update transaction
            transaction.Status = TransactionStatus.RolledBack;
            transaction.CompletedAt = DateTime.UtcNow;

            _logger.LogInformation("Transaction rolled back: {TransactionId}", transactionId);
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error rolling back transaction: {TransactionId}", transactionId);
            return false;
        }
    }

    /// <summary>
    /// Records an operation in a transaction
    /// </summary>
    /// <param name="transactionId">The transaction ID</param>
    /// <param name="type">The operation type</param>
    /// <param name="target">The operation target</param>
    /// <param name="details">The operation details</param>
    /// <returns>True if the operation was recorded successfully, false otherwise</returns>
    public bool RecordOperation(string transactionId, OperationType type, string target, string details)
    {
        try
        {
            _logger.LogInformation("Recording operation: {Type} on {Target}", type, target);

            // Check if transaction exists
            if (!_transactions.TryGetValue(transactionId, out var transaction))
            {
                _logger.LogWarning("Transaction not found: {TransactionId}", transactionId);
                return false;
            }

            // Check if transaction is active
            if (transaction.Status != TransactionStatus.Active)
            {
                _logger.LogWarning("Transaction is not active: {TransactionId}, Status: {Status}", transactionId, transaction.Status);
                return false;
            }

            // Create operation
            var operation = new Operation
            {
                Id = Guid.NewGuid().ToString(),
                Type = type,
                Target = target,
                Details = details,
                Timestamp = DateTime.UtcNow
            };

            // Add operation to transaction
            transaction.Operations.Add(operation);

            _logger.LogInformation("Operation recorded: {OperationId}", operation.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error recording operation: {Type} on {Target}", type, target);
            return false;
        }
    }

    /// <summary>
    /// Gets a transaction
    /// </summary>
    /// <param name="transactionId">The transaction ID</param>
    /// <returns>The transaction, or null if not found</returns>
    public Transaction? GetTransaction(string transactionId)
    {
        if (_transactions.TryGetValue(transactionId, out var transaction))
        {
            return transaction;
        }
        return null;
    }

    /// <summary>
    /// Gets all transactions for a context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The list of transactions</returns>
    public List<Transaction> GetTransactions(string contextId)
    {
        return _transactions.Values.Where(t => t.ContextId == contextId).ToList();
    }

    /// <summary>
    /// Gets all active transactions for a context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The list of active transactions</returns>
    public List<Transaction> GetActiveTransactions(string contextId)
    {
        return _transactions.Values.Where(t => t.ContextId == contextId && t.Status == TransactionStatus.Active).ToList();
    }

    /// <summary>
    /// Gets the available transaction options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "AutoCommit", "Whether to automatically commit transactions (true, false)" },
            { "AutoRollback", "Whether to automatically roll back transactions on error (true, false)" }
        };
    }
}
