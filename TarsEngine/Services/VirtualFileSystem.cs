using System.Text;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Provides a virtual file system for safe execution
/// </summary>
public class VirtualFileSystem
{
    private readonly ILogger<VirtualFileSystem> _logger;
    private readonly PermissionManager _permissionManager;
    private readonly Dictionary<string, VirtualFileSystemContext> _contexts = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="VirtualFileSystem"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="permissionManager">The permission manager</param>
    public VirtualFileSystem(ILogger<VirtualFileSystem> logger, PermissionManager permissionManager)
    {
        _logger = logger;
        _permissionManager = permissionManager;
    }

    /// <summary>
    /// Creates a virtual file system context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="workingDirectory">The working directory</param>
    /// <param name="backupDirectory">The backup directory</param>
    /// <param name="isDryRun">Whether the context is in dry run mode</param>
    /// <returns>The virtual file system context</returns>
    public async Task<VirtualFileSystemContext> CreateContextAsync(
        string contextId,
        string workingDirectory,
        string backupDirectory,
        bool isDryRun)
    {
        try
        {
            _logger.LogInformation("Creating virtual file system context: {ContextId}", contextId);

            // Create directories asynchronously
            await Task.WhenAll(
                Task.Run(() => Directory.CreateDirectory(workingDirectory)),
                Task.Run(() => Directory.CreateDirectory(backupDirectory))
            );

            // Create context
            var context = new VirtualFileSystemContext
            {
                ContextId = contextId,
                WorkingDirectory = workingDirectory,
                BackupDirectory = backupDirectory,
                IsDryRun = isDryRun,
                CreatedAt = DateTime.UtcNow
            };

            // Store context
            _contexts[contextId] = context;

            _logger.LogInformation("Created virtual file system context: {ContextId}", contextId);
            return context;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating virtual file system context: {ContextId}", contextId);
            throw;
        }
    }

    /// <summary>
    /// Gets a virtual file system context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The virtual file system context, or null if not found</returns>
    public VirtualFileSystemContext? GetContext(string contextId)
    {
        if (_contexts.TryGetValue(contextId, out var context))
        {
            return context;
        }

        _logger.LogWarning("Virtual file system context not found: {ContextId}", contextId);
        return null;
    }

    /// <summary>
    /// Removes a virtual file system context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    public void RemoveContext(string contextId)
    {
        if (_contexts.Remove(contextId))
        {
            _logger.LogInformation("Removed virtual file system context: {ContextId}", contextId);
        }
        else
        {
            _logger.LogWarning("Virtual file system context not found: {ContextId}", contextId);
        }
    }

    /// <summary>
    /// Reads a file
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>The file content</returns>
    public async Task<string> ReadFileAsync(string contextId, string filePath)
    {
        try
        {
            // Validate operation
            if (!_permissionManager.ValidateFileOperation(contextId, filePath, FileOperation.Read))
            {
                throw new UnauthorizedAccessException($"File read operation not allowed: {filePath}");
            }

            // Get context
            var context = GetContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Virtual file system context not found: {contextId}");
            }

            // Check if file exists in virtual file system
            if (context.VirtualFiles.TryGetValue(filePath, out var virtualFile))
            {
                _logger.LogInformation("Reading virtual file: {FilePath}", filePath);
                return virtualFile.Content;
            }

            // Check if file exists on disk
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            // Read file from disk
            _logger.LogInformation("Reading file from disk: {FilePath}", filePath);
            var content = await File.ReadAllTextAsync(filePath);

            // Add to virtual file system
            context.VirtualFiles[filePath] = new VirtualFile
            {
                Path = filePath,
                Content = content,
                OriginalContent = content,
                IsModified = false,
                IsCreated = false,
                IsDeleted = false
            };

            return content;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading file: {FilePath}", filePath);
            throw;
        }
    }

    /// <summary>
    /// Writes a file
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <returns>True if the file was written successfully, false otherwise</returns>
    public async Task<bool> WriteFileAsync(string contextId, string filePath, string content)
    {
        try
        {
            // Determine operation type
            var operation = File.Exists(filePath) ? FileOperation.Modify : FileOperation.Create;

            // Validate operation
            if (!_permissionManager.ValidateFileOperation(contextId, filePath, operation))
            {
                throw new UnauthorizedAccessException($"File write operation not allowed: {filePath}");
            }

            // Get context
            var context = GetContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Virtual file system context not found: {contextId}");
            }

            // Check file size limit
            if (!_permissionManager.EnforceFileSizeLimit(contextId, Encoding.UTF8.GetByteCount(content)))
            {
                throw new InvalidOperationException($"File size exceeds limit: {filePath}");
            }

            // Check file count limit
            if (operation == FileOperation.Create && !_permissionManager.EnforceFileCountLimit(contextId, context.VirtualFiles.Count + 1))
            {
                throw new InvalidOperationException($"File count exceeds limit");
            }

            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Check if file exists in virtual file system
            if (context.VirtualFiles.TryGetValue(filePath, out var virtualFile))
            {
                // Update virtual file
                virtualFile.Content = content;
                virtualFile.IsModified = true;

                // Add to modified files if not already there
                if (!context.ModifiedFiles.Contains(filePath))
                {
                    context.ModifiedFiles.Add(filePath);
                }
            }
            else
            {
                // Create backup if file exists on disk
                if (operation == FileOperation.Modify)
                {
                    await BackupFileAsync(contextId, filePath);
                }

                // Create virtual file
                context.VirtualFiles[filePath] = new VirtualFile
                {
                    Path = filePath,
                    Content = content,
                    OriginalContent = operation == FileOperation.Modify ? await File.ReadAllTextAsync(filePath) : string.Empty,
                    IsModified = operation == FileOperation.Modify,
                    IsCreated = operation == FileOperation.Create,
                    IsDeleted = false
                };

                // Add to modified or created files
                if (operation == FileOperation.Modify)
                {
                    context.ModifiedFiles.Add(filePath);
                }
                else
                {
                    context.CreatedFiles.Add(filePath);
                }
            }

            // Write file to disk if not in dry run mode
            if (!context.IsDryRun)
            {
                _logger.LogInformation("Writing file to disk: {FilePath}", filePath);
                await File.WriteAllTextAsync(filePath, content);
            }
            else
            {
                _logger.LogInformation("Dry run: Would write file to disk: {FilePath}", filePath);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error writing file: {FilePath}", filePath);
            throw;
        }
    }

    /// <summary>
    /// Deletes a file
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the file was deleted successfully, false otherwise</returns>
    public async Task<bool> DeleteFileAsync(string contextId, string filePath)
    {
        try
        {
            // Validate operation
            if (!_permissionManager.ValidateFileOperation(contextId, filePath, FileOperation.Delete))
            {
                throw new UnauthorizedAccessException($"File delete operation not allowed: {filePath}");
            }

            // Get context
            var context = GetContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Virtual file system context not found: {contextId}");
            }

            // Check if file exists
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            // Create backup
            await BackupFileAsync(contextId, filePath);

            // Check if file exists in virtual file system
            if (context.VirtualFiles.TryGetValue(filePath, out var virtualFile))
            {
                // Mark as deleted
                virtualFile.IsDeleted = true;
            }
            else
            {
                // Create virtual file and mark as deleted
                context.VirtualFiles[filePath] = new VirtualFile
                {
                    Path = filePath,
                    Content = string.Empty,
                    OriginalContent = await File.ReadAllTextAsync(filePath),
                    IsModified = false,
                    IsCreated = false,
                    IsDeleted = true
                };
            }

            // Add to deleted files
            if (!context.DeletedFiles.Contains(filePath))
            {
                context.DeletedFiles.Add(filePath);
            }

            // Delete file from disk if not in dry run mode
            if (!context.IsDryRun)
            {
                _logger.LogInformation("Deleting file from disk: {FilePath}", filePath);
                File.Delete(filePath);
            }
            else
            {
                _logger.LogInformation("Dry run: Would delete file from disk: {FilePath}", filePath);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting file: {FilePath}", filePath);
            throw;
        }
    }

    /// <summary>
    /// Backs up a file
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>The backup file path</returns>
    public async Task<string> BackupFileAsync(string contextId, string filePath)
    {
        try
        {
            // Get context
            var context = GetContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Virtual file system context not found: {contextId}");
            }

            // Check if file exists
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            // Check if file is already backed up
            if (context.BackedUpFiles.Contains(filePath))
            {
                _logger.LogInformation("File already backed up: {FilePath}", filePath);
                return GetBackupFilePath(context, filePath);
            }

            // Create backup directory if it doesn't exist
            if (!Directory.Exists(context.BackupDirectory))
            {
                Directory.CreateDirectory(context.BackupDirectory);
            }

            // Create backup file path
            var backupFilePath = GetBackupFilePath(context, filePath);

            // Create directory structure in backup directory
            var backupDirectory = Path.GetDirectoryName(backupFilePath);
            if (!string.IsNullOrEmpty(backupDirectory) && !Directory.Exists(backupDirectory))
            {
                Directory.CreateDirectory(backupDirectory);
            }

            // Copy file to backup
            _logger.LogInformation("Backing up file: {FilePath} to {BackupFilePath}", filePath, backupFilePath);
            await Task.Run(() => File.Copy(filePath, backupFilePath, true));

            // Add to backed up files
            context.BackedUpFiles.Add(filePath);

            return backupFilePath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error backing up file: {FilePath}", filePath);
            throw;
        }
    }

    /// <summary>
    /// Restores a file from backup
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the file was restored successfully, false otherwise</returns>
    public async Task<bool> RestoreFileAsync(string contextId, string filePath)
    {
        try
        {
            // Get context
            var context = GetContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Virtual file system context not found: {contextId}");
            }

            // Check if file is backed up
            if (!context.BackedUpFiles.Contains(filePath))
            {
                throw new InvalidOperationException($"File not backed up: {filePath}");
            }

            // Get backup file path
            var backupFilePath = GetBackupFilePath(context, filePath);

            // Check if backup file exists
            if (!File.Exists(backupFilePath))
            {
                throw new FileNotFoundException($"Backup file not found: {backupFilePath}");
            }

            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Copy backup file to original location
            _logger.LogInformation("Restoring file: {FilePath} from {BackupFilePath}", filePath, backupFilePath);
            await Task.Run(() => File.Copy(backupFilePath, filePath, true));

            // Update virtual file system
            if (context.VirtualFiles.TryGetValue(filePath, out var virtualFile))
            {
                virtualFile.Content = virtualFile.OriginalContent;
                virtualFile.IsModified = false;
                virtualFile.IsCreated = false;
                virtualFile.IsDeleted = false;
            }

            // Remove from modified, created, and deleted files
            context.ModifiedFiles.Remove(filePath);
            context.CreatedFiles.Remove(filePath);
            context.DeletedFiles.Remove(filePath);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error restoring file: {FilePath}", filePath);
            throw;
        }
    }

    /// <summary>
    /// Commits changes to the file system
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>True if the changes were committed successfully, false otherwise</returns>
    public async Task<bool> CommitChangesAsync(string contextId)
    {
        try
        {
            // Get context
            var context = GetContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Virtual file system context not found: {contextId}");
            }

            // Check if in dry run mode
            if (context.IsDryRun)
            {
                _logger.LogInformation("Dry run: Would commit changes for context: {ContextId}", contextId);
                return true;
            }

            _logger.LogInformation("Committing changes for context: {ContextId}", contextId);

            // Apply changes to disk
            foreach (var virtualFile in context.VirtualFiles.Values)
            {
                if (virtualFile.IsDeleted)
                {
                    // Delete file
                    if (File.Exists(virtualFile.Path))
                    {
                        _logger.LogInformation("Deleting file: {FilePath}", virtualFile.Path);
                        File.Delete(virtualFile.Path);
                    }
                }
                else if (virtualFile.IsCreated || virtualFile.IsModified)
                {
                    // Create directory if it doesn't exist
                    var directory = Path.GetDirectoryName(virtualFile.Path);
                    if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                    {
                        Directory.CreateDirectory(directory);
                    }

                    // Write file
                    _logger.LogInformation("{Operation} file: {FilePath}", virtualFile.IsCreated ? "Creating" : "Modifying", virtualFile.Path);
                    await File.WriteAllTextAsync(virtualFile.Path, virtualFile.Content);
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error committing changes for context: {ContextId}", contextId);
            throw;
        }
    }

    /// <summary>
    /// Rolls back changes to the file system
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>True if the changes were rolled back successfully, false otherwise</returns>
    public async Task<bool> RollbackChangesAsync(string contextId)
    {
        try
        {
            // Get context
            var context = GetContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Virtual file system context not found: {contextId}");
            }

            // Check if in dry run mode
            if (context.IsDryRun)
            {
                _logger.LogInformation("Dry run: Would roll back changes for context: {ContextId}", contextId);
                return true;
            }

            _logger.LogInformation("Rolling back changes for context: {ContextId}", contextId);

            // Restore backed up files
            foreach (var filePath in context.BackedUpFiles)
            {
                await RestoreFileAsync(contextId, filePath);
            }

            // Delete created files
            foreach (var filePath in context.CreatedFiles)
            {
                if (File.Exists(filePath))
                {
                    _logger.LogInformation("Deleting created file: {FilePath}", filePath);
                    File.Delete(filePath);
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error rolling back changes for context: {ContextId}", contextId);
            throw;
        }
    }

    /// <summary>
    /// Gets the list of modified files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The list of modified files</returns>
    public List<string> GetModifiedFiles(string contextId)
    {
        var context = GetContext(contextId);
        return context?.ModifiedFiles.ToList() ?? new List<string>();
    }

    /// <summary>
    /// Gets the list of created files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The list of created files</returns>
    public List<string> GetCreatedFiles(string contextId)
    {
        var context = GetContext(contextId);
        return context?.CreatedFiles.ToList() ?? new List<string>();
    }

    /// <summary>
    /// Gets the list of deleted files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The list of deleted files</returns>
    public List<string> GetDeletedFiles(string contextId)
    {
        var context = GetContext(contextId);
        return context?.DeletedFiles.ToList() ?? new List<string>();
    }

    /// <summary>
    /// Gets the list of backed up files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The list of backed up files</returns>
    public List<string> GetBackedUpFiles(string contextId)
    {
        var context = GetContext(contextId);
        return context?.BackedUpFiles.ToList() ?? new List<string>();
    }

    /// <summary>
    /// Gets the backup file path for a file
    /// </summary>
    /// <param name="context">The virtual file system context</param>
    /// <param name="filePath">The file path</param>
    /// <returns>The backup file path</returns>
    private string GetBackupFilePath(VirtualFileSystemContext context, string filePath)
    {
        // Create a relative path from the root
        var relativePath = Path.GetFullPath(filePath).Replace(Path.GetPathRoot(filePath) ?? string.Empty, string.Empty);

        // Combine with backup directory
        return Path.Combine(context.BackupDirectory, relativePath);
    }

    /// <summary>
    /// Gets the available file system options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "WorkingDirectory", "The working directory for the virtual file system" },
            { "BackupDirectory", "The backup directory for the virtual file system" },
            { "IsDryRun", "Whether the virtual file system is in dry run mode (default: true)" }
        };
    }
}
