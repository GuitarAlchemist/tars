using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Provides backup functionality for files
/// </summary>
public class FileBackupService
{
    private readonly ILogger<FileBackupService> _logger;
    private readonly Dictionary<string, Dictionary<string, FileBackupInfo>> _backups = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="FileBackupService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public FileBackupService(ILogger<FileBackupService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Creates a backup context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="backupDirectory">The backup directory</param>
    /// <returns>True if the backup context was created successfully, false otherwise</returns>
    public bool CreateBackupContext(string contextId, string backupDirectory)
    {
        try
        {
            _logger.LogInformation("Creating backup context: {ContextId}", contextId);

            // Check if context already exists
            if (_backups.ContainsKey(contextId))
            {
                _logger.LogWarning("Backup context already exists: {ContextId}", contextId);
                return false;
            }

            // Create backup directory if it doesn't exist
            if (!Directory.Exists(backupDirectory))
            {
                Directory.CreateDirectory(backupDirectory);
            }

            // Create backup context
            _backups[contextId] = new Dictionary<string, FileBackupInfo>();

            _logger.LogInformation("Created backup context: {ContextId}", contextId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating backup context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Removes a backup context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="deleteBackupFiles">Whether to delete backup files</param>
    /// <returns>True if the backup context was removed successfully, false otherwise</returns>
    public bool RemoveBackupContext(string contextId, bool deleteBackupFiles = false)
    {
        try
        {
            _logger.LogInformation("Removing backup context: {ContextId}", contextId);

            // Check if context exists
            if (!_backups.TryGetValue(contextId, out var backupInfo))
            {
                _logger.LogWarning("Backup context not found: {ContextId}", contextId);
                return false;
            }

            // Delete backup files if requested
            if (deleteBackupFiles)
            {
                foreach (var backup in backupInfo.Values)
                {
                    if (File.Exists(backup.BackupFilePath))
                    {
                        File.Delete(backup.BackupFilePath);
                    }
                }
            }

            // Remove backup context
            _backups.Remove(contextId);

            _logger.LogInformation("Removed backup context: {ContextId}", contextId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing backup context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Creates a backup of a file
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="filePath">The file path</param>
    /// <param name="backupDirectory">The backup directory</param>
    /// <returns>The backup file path, or null if the backup failed</returns>
    public async Task<string?> BackupFileAsync(string contextId, string filePath, string backupDirectory)
    {
        try
        {
            _logger.LogInformation("Backing up file: {FilePath}", filePath);

            // Check if context exists
            if (!_backups.TryGetValue(contextId, out var backupInfo))
            {
                _logger.LogWarning("Backup context not found: {ContextId}", contextId);
                return null;
            }

            // Check if file exists
            if (!File.Exists(filePath))
            {
                _logger.LogWarning("File not found: {FilePath}", filePath);
                return null;
            }

            // Check if file is already backed up
            if (backupInfo.TryGetValue(filePath, out var existingBackup))
            {
                _logger.LogInformation("File already backed up: {FilePath}", filePath);
                return existingBackup.BackupFilePath;
            }

            // Create backup directory if it doesn't exist
            if (!Directory.Exists(backupDirectory))
            {
                Directory.CreateDirectory(backupDirectory);
            }

            // Create backup file path
            var backupFilePath = GetBackupFilePath(filePath, backupDirectory);

            // Create directory structure in backup directory
            var backupFileDirectory = Path.GetDirectoryName(backupFilePath);
            if (!string.IsNullOrEmpty(backupFileDirectory) && !Directory.Exists(backupFileDirectory))
            {
                Directory.CreateDirectory(backupFileDirectory);
            }

            // Copy file to backup
            await Task.Run(() => File.Copy(filePath, backupFilePath, true));

            // Calculate file hash
            var fileHash = await CalculateFileHashAsync(filePath);

            // Create backup info
            var backup = new FileBackupInfo
            {
                FilePath = filePath,
                BackupFilePath = backupFilePath,
                BackupTime = DateTime.UtcNow,
                FileHash = fileHash
            };

            // Add to backup info
            backupInfo[filePath] = backup;

            _logger.LogInformation("Backed up file: {FilePath} to {BackupFilePath}", filePath, backupFilePath);
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
    /// <param name="filePath">The file path</param>
    /// <returns>True if the file was restored successfully, false otherwise</returns>
    public async Task<bool> RestoreFileAsync(string contextId, string filePath)
    {
        try
        {
            _logger.LogInformation("Restoring file: {FilePath}", filePath);

            // Check if context exists
            if (!_backups.TryGetValue(contextId, out var backupInfo))
            {
                _logger.LogWarning("Backup context not found: {ContextId}", contextId);
                return false;
            }

            // Check if file is backed up
            if (!backupInfo.TryGetValue(filePath, out var backup))
            {
                _logger.LogWarning("File not backed up: {FilePath}", filePath);
                return false;
            }

            // Check if backup file exists
            if (!File.Exists(backup.BackupFilePath))
            {
                _logger.LogWarning("Backup file not found: {BackupFilePath}", backup.BackupFilePath);
                return false;
            }

            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Copy backup file to original location
            await Task.Run(() => File.Copy(backup.BackupFilePath, filePath, true));

            // Verify file hash
            var fileHash = await CalculateFileHashAsync(filePath);
            if (fileHash != backup.FileHash)
            {
                _logger.LogWarning("File hash mismatch after restore: {FilePath}", filePath);
            }

            _logger.LogInformation("Restored file: {FilePath} from {BackupFilePath}", filePath, backup.BackupFilePath);
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
            if (!_backups.TryGetValue(contextId, out var backupInfo))
            {
                _logger.LogWarning("Backup context not found: {ContextId}", contextId);
                return false;
            }

            // Restore each file
            var success = true;
            foreach (var filePath in backupInfo.Keys.ToList())
            {
                var result = await RestoreFileAsync(contextId, filePath);
                if (!result)
                {
                    success = false;
                }
            }

            _logger.LogInformation("Restored all files for context: {ContextId}", contextId);
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error restoring all files for context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Gets the backup file path
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="backupDirectory">The backup directory</param>
    /// <returns>The backup file path</returns>
    private string GetBackupFilePath(string filePath, string backupDirectory)
    {
        // Create a relative path from the root
        var relativePath = Path.GetFullPath(filePath).Replace(Path.GetPathRoot(filePath) ?? string.Empty, string.Empty);

        // Combine with backup directory
        return Path.Combine(backupDirectory, relativePath);
    }

    /// <summary>
    /// Calculates the hash of a file
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <returns>The file hash</returns>
    private async Task<string> CalculateFileHashAsync(string filePath)
    {
        using var md5 = MD5.Create();
        using var stream = File.OpenRead(filePath);
        var hashBytes = await md5.ComputeHashAsync(stream);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }

    /// <summary>
    /// Gets the list of backed up files
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The list of backed up files</returns>
    public List<string> GetBackedUpFiles(string contextId)
    {
        if (_backups.TryGetValue(contextId, out var backupInfo))
        {
            return backupInfo.Keys.ToList();
        }
        return new List<string>();
    }

    /// <summary>
    /// Gets the backup info for a file
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>The backup info, or null if not found</returns>
    public FileBackupInfo? GetBackupInfo(string contextId, string filePath)
    {
        if (_backups.TryGetValue(contextId, out var backupInfo) && backupInfo.TryGetValue(filePath, out var backup))
        {
            return backup;
        }
        return null;
    }

    /// <summary>
    /// Gets all backup info for a context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The dictionary of backup info</returns>
    public Dictionary<string, FileBackupInfo> GetAllBackupInfo(string contextId)
    {
        if (_backups.TryGetValue(contextId, out var backupInfo))
        {
            return new Dictionary<string, FileBackupInfo>(backupInfo);
        }
        return new Dictionary<string, FileBackupInfo>();
    }

    /// <summary>
    /// Checks if a file is backed up
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the file is backed up, false otherwise</returns>
    public bool IsFileBackedUp(string contextId, string filePath)
    {
        return _backups.TryGetValue(contextId, out var backupInfo) && backupInfo.ContainsKey(filePath);
    }

    /// <summary>
    /// Gets the available backup options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "BackupDirectory", "The directory where backups are stored" },
            { "DeleteBackupFiles", "Whether to delete backup files when removing a context (true, false)" }
        };
    }
}
