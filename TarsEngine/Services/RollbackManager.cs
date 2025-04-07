using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Manages rollbacks for changes
/// </summary>
public class RollbackManager
{
    private readonly ILogger<RollbackManager> _logger;
    private readonly FileBackupService _fileBackupService;
    private readonly TransactionManager _transactionManager;
    private readonly AuditTrailService _auditTrailService;
    private readonly Dictionary<string, RollbackContext> _rollbackContexts = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="RollbackManager"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="fileBackupService">The file backup service</param>
    /// <param name="transactionManager">The transaction manager</param>
    /// <param name="auditTrailService">The audit trail service</param>
    public RollbackManager(
        ILogger<RollbackManager> logger,
        FileBackupService fileBackupService,
        TransactionManager transactionManager,
        AuditTrailService auditTrailService)
    {
        _logger = logger;
        _fileBackupService = fileBackupService;
        _transactionManager = transactionManager;
        _auditTrailService = auditTrailService;
    }

    /// <summary>
    /// Creates a rollback context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="userId">The user ID</param>
    /// <param name="description">The context description</param>
    /// <param name="backupDirectory">The backup directory</param>
    /// <returns>True if the rollback context was created successfully, false otherwise</returns>
    public bool CreateRollbackContext(string contextId, string userId, string description, string backupDirectory)
    {
        try
        {
            _logger.LogInformation("Creating rollback context: {ContextId}", contextId);

            // Check if context already exists
            if (_rollbackContexts.ContainsKey(contextId))
            {
                _logger.LogWarning("Rollback context already exists: {ContextId}", contextId);
                return false;
            }

            // Create backup context
            var backupResult = _fileBackupService.CreateBackupContext(contextId, backupDirectory);
            if (!backupResult)
            {
                _logger.LogWarning("Failed to create backup context: {ContextId}", contextId);
                return false;
            }

            // Create audit context
            var auditResult = _auditTrailService.CreateAuditContext(contextId, userId, description);
            if (!auditResult)
            {
                _logger.LogWarning("Failed to create audit context: {ContextId}", contextId);
                _fileBackupService.RemoveBackupContext(contextId);
                return false;
            }

            // Create rollback context
            var rollbackContext = new RollbackContext
            {
                ContextId = contextId,
                UserId = userId,
                Description = description,
                BackupDirectory = backupDirectory,
                CreatedAt = DateTime.UtcNow
            };

            // Store rollback context
            _rollbackContexts[contextId] = rollbackContext;

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.ContextCreation,
                "RollbackContext",
                contextId,
                userId,
                $"Created rollback context: {description}");

            _logger.LogInformation("Created rollback context: {ContextId}", contextId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating rollback context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Removes a rollback context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="deleteBackupFiles">Whether to delete backup files</param>
    /// <returns>True if the rollback context was removed successfully, false otherwise</returns>
    public bool RemoveRollbackContext(string contextId, bool deleteBackupFiles = false)
    {
        try
        {
            _logger.LogInformation("Removing rollback context: {ContextId}", contextId);

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return false;
            }

            // Remove backup context
            _fileBackupService.RemoveBackupContext(contextId, deleteBackupFiles);

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.ContextCompletion,
                "RollbackContext",
                contextId,
                rollbackContext.UserId,
                $"Removed rollback context: {rollbackContext.Description}");

            // Remove rollback context
            _rollbackContexts.Remove(contextId);

            _logger.LogInformation("Removed rollback context: {ContextId}", contextId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing rollback context: {ContextId}", contextId);
            return false;
        }
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

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return string.Empty;
            }

            // Begin transaction
            var transactionId = _transactionManager.BeginTransaction(contextId, name, description);
            if (string.IsNullOrEmpty(transactionId))
            {
                _logger.LogWarning("Failed to begin transaction: {Name}", name);
                return string.Empty;
            }

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.TransactionBegin,
                "Transaction",
                transactionId,
                rollbackContext.UserId,
                $"Began transaction: {name}",
                description);

            _logger.LogInformation("Transaction begun: {TransactionId}", transactionId);
            return transactionId;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error beginning transaction: {Name}", name);
            return string.Empty;
        }
    }

    /// <summary>
    /// Commits a transaction
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="transactionId">The transaction ID</param>
    /// <returns>True if the transaction was committed successfully, false otherwise</returns>
    public bool CommitTransaction(string contextId, string transactionId)
    {
        try
        {
            _logger.LogInformation("Committing transaction: {TransactionId}", transactionId);

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return false;
            }

            // Commit transaction
            var result = _transactionManager.CommitTransaction(transactionId);
            if (!result)
            {
                _logger.LogWarning("Failed to commit transaction: {TransactionId}", transactionId);
                return false;
            }

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.TransactionCommit,
                "Transaction",
                transactionId,
                rollbackContext.UserId,
                $"Committed transaction: {transactionId}");

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
    /// <param name="contextId">The context ID</param>
    /// <param name="transactionId">The transaction ID</param>
    /// <returns>True if the transaction was rolled back successfully, false otherwise</returns>
    public async Task<bool> RollbackTransactionAsync(string contextId, string transactionId)
    {
        try
        {
            _logger.LogInformation("Rolling back transaction: {TransactionId}", transactionId);

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return false;
            }

            // Rollback transaction
            var result = await _transactionManager.RollbackTransactionAsync(transactionId);
            if (!result)
            {
                _logger.LogWarning("Failed to roll back transaction: {TransactionId}", transactionId);
                return false;
            }

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.TransactionRollback,
                "Transaction",
                transactionId,
                rollbackContext.UserId,
                $"Rolled back transaction: {transactionId}");

            _logger.LogInformation("Transaction rolled back: {TransactionId}", transactionId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error rolling back transaction: {TransactionId}", transactionId);
            return false;
        }
    }

    /// <summary>
    /// Backs up a file
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="transactionId">The transaction ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>The backup file path, or null if the backup failed</returns>
    public async Task<string?> BackupFileAsync(string contextId, string transactionId, string filePath)
    {
        try
        {
            _logger.LogInformation("Backing up file: {FilePath}", filePath);

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return null;
            }

            // Check if file exists
            if (!File.Exists(filePath))
            {
                _logger.LogWarning("File not found: {FilePath}", filePath);
                return null;
            }

            // Backup file
            var backupFilePath = await _fileBackupService.BackupFileAsync(contextId, filePath, rollbackContext.BackupDirectory);
            if (backupFilePath == null)
            {
                _logger.LogWarning("Failed to backup file: {FilePath}", filePath);
                return null;
            }

            // Record operation
            if (!string.IsNullOrEmpty(transactionId))
            {
                _transactionManager.RecordOperation(
                    transactionId,
                    OperationType.FileBackup,
                    filePath,
                    $"Backed up file: {filePath} to {backupFilePath}");
            }

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.FileBackup,
                "File",
                filePath,
                rollbackContext.UserId,
                $"Backed up file: {filePath}",
                $"Backup file path: {backupFilePath}");

            _logger.LogInformation("File backed up: {FilePath} to {BackupFilePath}", filePath, backupFilePath);
            return backupFilePath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error backing up file: {FilePath}", filePath);
            return null;
        }
    }

    /// <summary>
    /// Restores a file from backup
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="transactionId">The transaction ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the file was restored successfully, false otherwise</returns>
    public async Task<bool> RestoreFileAsync(string contextId, string transactionId, string filePath)
    {
        try
        {
            _logger.LogInformation("Restoring file: {FilePath}", filePath);

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return false;
            }

            // Restore file
            var result = await _fileBackupService.RestoreFileAsync(contextId, filePath);
            if (!result)
            {
                _logger.LogWarning("Failed to restore file: {FilePath}", filePath);
                return false;
            }

            // Record operation
            if (!string.IsNullOrEmpty(transactionId))
            {
                _transactionManager.RecordOperation(
                    transactionId,
                    OperationType.FileRestore,
                    filePath,
                    $"Restored file: {filePath}");
            }

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.FileRestore,
                "File",
                filePath,
                rollbackContext.UserId,
                $"Restored file: {filePath}");

            _logger.LogInformation("File restored: {FilePath}", filePath);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error restoring file: {FilePath}", filePath);
            return false;
        }
    }

    /// <summary>
    /// Restores all files from backup
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>True if all files were restored successfully, false otherwise</returns>
    public async Task<bool> RestoreAllFilesAsync(string contextId)
    {
        try
        {
            _logger.LogInformation("Restoring all files for context: {ContextId}", contextId);

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return false;
            }

            // Restore all files
            var result = await _fileBackupService.RestoreAllFilesAsync(contextId);
            if (!result)
            {
                _logger.LogWarning("Failed to restore all files for context: {ContextId}", contextId);
                return false;
            }

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                AuditEntryType.FileRestore,
                "Context",
                contextId,
                rollbackContext.UserId,
                $"Restored all files for context: {contextId}");

            _logger.LogInformation("All files restored for context: {ContextId}", contextId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error restoring all files for context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Records a file operation
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="transactionId">The transaction ID</param>
    /// <param name="type">The operation type</param>
    /// <param name="filePath">The file path</param>
    /// <param name="details">The operation details</param>
    /// <returns>True if the operation was recorded successfully, false otherwise</returns>
    public bool RecordFileOperation(
        string contextId,
        string transactionId,
        OperationType type,
        string filePath,
        string details)
    {
        try
        {
            _logger.LogInformation("Recording file operation: {Type} on {FilePath}", type, filePath);

            // Check if context exists
            if (!_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
            {
                _logger.LogWarning("Rollback context not found: {ContextId}", contextId);
                return false;
            }

            // Record operation
            if (!string.IsNullOrEmpty(transactionId))
            {
                var result = _transactionManager.RecordOperation(transactionId, type, filePath, details);
                if (!result)
                {
                    _logger.LogWarning("Failed to record operation: {Type} on {FilePath}", type, filePath);
                    return false;
                }
            }

            // Map operation type to audit entry type
            var auditEntryType = type switch
            {
                OperationType.FileCreation => AuditEntryType.FileCreation,
                OperationType.FileModification => AuditEntryType.FileModification,
                OperationType.FileDeletion => AuditEntryType.FileDeletion,
                OperationType.FileBackup => AuditEntryType.FileBackup,
                OperationType.FileRestore => AuditEntryType.FileRestore,
                OperationType.CommandExecution => AuditEntryType.CommandExecution,
                _ => AuditEntryType.Other
            };

            // Record audit entry
            _auditTrailService.RecordEntry(
                contextId,
                auditEntryType,
                "File",
                filePath,
                rollbackContext.UserId,
                $"{type} on {filePath}",
                details);

            _logger.LogInformation("File operation recorded: {Type} on {FilePath}", type, filePath);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error recording file operation: {Type} on {FilePath}", type, filePath);
            return false;
        }
    }

    /// <summary>
    /// Gets a rollback context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The rollback context, or null if not found</returns>
    public RollbackContext? GetRollbackContext(string contextId)
    {
        if (_rollbackContexts.TryGetValue(contextId, out var rollbackContext))
        {
            return rollbackContext;
        }
        return null;
    }

    /// <summary>
    /// Gets all rollback contexts
    /// </summary>
    /// <returns>The list of rollback contexts</returns>
    public List<RollbackContext> GetRollbackContexts()
    {
        return _rollbackContexts.Values.ToList();
    }

    /// <summary>
    /// Gets the list of backed up files
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The list of backed up files</returns>
    public List<string> GetBackedUpFiles(string contextId)
    {
        return _fileBackupService.GetBackedUpFiles(contextId);
    }

    /// <summary>
    /// Gets all transactions for a context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The list of transactions</returns>
    public List<Transaction> GetTransactions(string contextId)
    {
        return _transactionManager.GetTransactions(contextId);
    }

    /// <summary>
    /// Gets all audit entries for a context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The list of audit entries</returns>
    public List<AuditEntry> GetAuditEntries(string contextId)
    {
        return _auditTrailService.GetAuditEntries(contextId);
    }

    /// <summary>
    /// Exports audit entries to a file
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the audit entries were exported successfully, false otherwise</returns>
    public async Task<bool> ExportAuditEntriesAsync(string contextId, string filePath)
    {
        return await _auditTrailService.ExportAuditEntriesAsync(contextId, filePath);
    }

    /// <summary>
    /// Gets the available rollback options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        var options = new Dictionary<string, string>
        {
            { "BackupDirectory", "The directory where backups are stored" },
            { "DeleteBackupFiles", "Whether to delete backup files when removing a context (true, false)" },
            { "AutoBackup", "Whether to automatically backup files before modification (true, false)" },
            { "AutoRollback", "Whether to automatically roll back on error (true, false)" }
        };

        // Add file backup options
        var fileBackupOptions = _fileBackupService.GetAvailableOptions();
        foreach (var option in fileBackupOptions)
        {
            options[option.Key] = option.Value;
        }

        // Add transaction options
        var transactionOptions = _transactionManager.GetAvailableOptions();
        foreach (var option in transactionOptions)
        {
            options[option.Key] = option.Value;
        }

        // Add audit options
        var auditOptions = _auditTrailService.GetAvailableOptions();
        foreach (var option in auditOptions)
        {
            options[option.Key] = option.Value;
        }

        return options;
    }
}
