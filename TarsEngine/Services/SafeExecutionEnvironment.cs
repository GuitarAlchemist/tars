using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Utilities;

namespace TarsEngine.Services;

/// <summary>
/// Provides a safe environment for executing changes
/// </summary>
public class SafeExecutionEnvironment
{
    private readonly ILogger<SafeExecutionEnvironment> _logger;
    private readonly PermissionManager _permissionManager;
    private readonly VirtualFileSystem _virtualFileSystem;
    private readonly Dictionary<string, TarsEngine.Models.ExecutionContext> _executionContexts = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="SafeExecutionEnvironment"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="permissionManager">The permission manager</param>
    /// <param name="virtualFileSystem">The virtual file system</param>
    public SafeExecutionEnvironment(
        ILogger<SafeExecutionEnvironment> logger,
        PermissionManager permissionManager,
        VirtualFileSystem virtualFileSystem)
    {
        _logger = logger;
        _permissionManager = permissionManager;
        _virtualFileSystem = virtualFileSystem;
    }

    /// <summary>
    /// Creates an execution context
    /// </summary>
    /// <param name="executionPlanId">The execution plan ID</param>
    /// <param name="improvementId">The improvement ID</param>
    /// <param name="metascriptId">The metascript ID</param>
    /// <param name="mode">The execution mode</param>
    /// <param name="environment">The execution environment</param>
    /// <param name="options">Optional context options</param>
    /// <returns>The execution context</returns>
    public async Task<TarsEngine.Models.ExecutionContext> CreateExecutionContextAsync(
        string executionPlanId,
        string improvementId,
        string metascriptId,
        ExecutionMode mode,
        ExecutionEnvironment environment,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Creating execution context for plan: {ExecutionPlanId}", executionPlanId);

            // Create execution context
            var context = new TarsEngine.Models.ExecutionContext
            {
                ExecutionPlanId = executionPlanId,
                ImprovementId = improvementId,
                MetascriptId = metascriptId,
                Mode = mode,
                Environment = environment,
                CreatedAt = DateTime.UtcNow
            };

            // Set working directory
            var workingDirectory = options?.TryGetValue("WorkingDirectory", out var wd) == true && !string.IsNullOrEmpty(wd)
                ? wd
                : Path.Combine(Path.GetTempPath(), "TARS", "Execution", executionPlanId);
            context.WorkingDirectory = workingDirectory;

            // Set backup directory
            var backupDirectory = options?.TryGetValue("BackupDirectory", out var bd) == true && !string.IsNullOrEmpty(bd)
                ? bd
                : Path.Combine(Path.GetTempPath(), "TARS", "Backup", executionPlanId);
            context.BackupDirectory = backupDirectory;

            // Set output directory
            var outputDirectory = options?.TryGetValue("OutputDirectory", out var od) == true && !string.IsNullOrEmpty(od)
                ? od
                : Path.Combine(Path.GetTempPath(), "TARS", "Output", executionPlanId);
            context.OutputDirectory = outputDirectory;

            // Create directories
            Directory.CreateDirectory(workingDirectory);
            Directory.CreateDirectory(backupDirectory);
            Directory.CreateDirectory(outputDirectory);

            // Set timeout
            if (options?.TryGetValue("TimeoutMs", out var timeoutStr) == true && int.TryParse(timeoutStr, out var timeout))
            {
                context.TimeoutMs = timeout;
            }

            // Set options
            if (options != null)
            {
                foreach (var option in options)
                {
                    context.Options[option.Key] = option.Value;
                }
            }

            // Create permissions
            var permissions = _permissionManager.CreatePermissions(context.Id, mode, environment, options);
            context.Permissions = permissions;

            // Create virtual file system context
            var vfsContext = _virtualFileSystem.CreateContext(context.Id, workingDirectory, backupDirectory, mode == ExecutionMode.DryRun);

            // Store execution context
            _executionContexts[context.Id] = context;

            _logger.LogInformation("Created execution context: {ContextId}", context.Id);
            return context;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating execution context for plan: {ExecutionPlanId}", executionPlanId);
            throw;
        }
    }

    /// <summary>
    /// Gets an execution context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The execution context, or null if not found</returns>
    public TarsEngine.Models.ExecutionContext? GetExecutionContext(string contextId)
    {
        if (_executionContexts.TryGetValue(contextId, out var context))
        {
            return context;
        }

        _logger.LogWarning("Execution context not found: {ContextId}", contextId);
        return null;
    }

    /// <summary>
    /// Removes an execution context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    public void RemoveExecutionContext(string contextId)
    {
        try
        {
            // Remove execution context
            if (_executionContexts.Remove(contextId))
            {
                _logger.LogInformation("Removed execution context: {ContextId}", contextId);
            }
            else
            {
                _logger.LogWarning("Execution context not found: {ContextId}", contextId);
            }

            // Remove virtual file system context
            _virtualFileSystem.RemoveContext(contextId);

            // Remove permissions
            _permissionManager.RemovePermissions(contextId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing execution context: {ContextId}", contextId);
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
            _logger.LogInformation("Reading file: {FilePath}", filePath);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), $"Reading file: {filePath}");
            context.AddAffectedFile(filePath);

            // Read file
            return await _virtualFileSystem.ReadFileAsync(contextId, filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading file: {FilePath}", filePath);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError($"Error reading file: {filePath}", "SafeExecutionEnvironment", ex);

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
            _logger.LogInformation("Writing file: {FilePath}", filePath);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), $"Writing file: {filePath}");

            // Write file
            var result = await _virtualFileSystem.WriteFileAsync(contextId, filePath, content);

            // Update context with file information
            if (File.Exists(filePath))
            {
                context.AddModifiedFile(filePath);
            }
            else
            {
                context.AddCreatedFile(filePath);
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error writing file: {FilePath}", filePath);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError($"Error writing file: {filePath}", "SafeExecutionEnvironment", ex);

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
            _logger.LogInformation("Deleting file: {FilePath}", filePath);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), $"Deleting file: {filePath}");

            // Delete file
            var result = await _virtualFileSystem.DeleteFileAsync(contextId, filePath);

            // Update context with file information
            context.AddDeletedFile(filePath);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting file: {FilePath}", filePath);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError($"Error deleting file: {filePath}", "SafeExecutionEnvironment", ex);

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
            _logger.LogInformation("Backing up file: {FilePath}", filePath);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), $"Backing up file: {filePath}");

            // Backup file
            var backupFilePath = await _virtualFileSystem.BackupFileAsync(contextId, filePath);

            // Update context with file information
            context.AddBackedUpFile(filePath);

            return backupFilePath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error backing up file: {FilePath}", filePath);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError($"Error backing up file: {filePath}", "SafeExecutionEnvironment", ex);

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
            _logger.LogInformation("Restoring file: {FilePath}", filePath);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), $"Restoring file: {filePath}");

            // Restore file
            return await _virtualFileSystem.RestoreFileAsync(contextId, filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error restoring file: {FilePath}", filePath);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError($"Error restoring file: {filePath}", "SafeExecutionEnvironment", ex);

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
            _logger.LogInformation("Committing changes for context: {ContextId}", contextId);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), "Committing changes");

            // Commit changes
            return await _virtualFileSystem.CommitChangesAsync(contextId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error committing changes for context: {ContextId}", contextId);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError("Error committing changes", "SafeExecutionEnvironment", ex);

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
            _logger.LogInformation("Rolling back changes for context: {ContextId}", contextId);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), "Rolling back changes");

            // Rollback changes
            return await _virtualFileSystem.RollbackChangesAsync(contextId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error rolling back changes for context: {ContextId}", contextId);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError("Error rolling back changes", "SafeExecutionEnvironment", ex);

            throw;
        }
    }

    /// <summary>
    /// Executes a command
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="command">The command to execute</param>
    /// <param name="arguments">The command arguments</param>
    /// <param name="workingDirectory">The working directory</param>
    /// <returns>The command output</returns>
    public async Task<string> ExecuteCommandAsync(string contextId, string command, string arguments, string? workingDirectory = null)
    {
        try
        {
            _logger.LogInformation("Executing command: {Command} {Arguments}", command, arguments);

            // Get execution context
            var context = GetExecutionContext(contextId);
            if (context == null)
            {
                throw new InvalidOperationException($"Execution context not found: {contextId}");
            }

            // Validate process execution
            if (!_permissionManager.ValidateProcessExecution(contextId, command))
            {
                throw new UnauthorizedAccessException($"Process execution not allowed: {command}");
            }

            // Update context
            context.UpdatedAt = DateTime.UtcNow;
            context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Information), $"Executing command: {command} {arguments}");

            // Set working directory
            workingDirectory ??= context.WorkingDirectory;

            // Check if in dry run mode
            if (context.IsDryRun)
            {
                _logger.LogInformation("Dry run: Would execute command: {Command} {Arguments}", command, arguments);
                return $"[Dry run] Command: {command} {arguments}";
            }

            // Execute command
            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = command,
                    Arguments = arguments,
                    WorkingDirectory = workingDirectory,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            // Check exit code
            if (process.ExitCode != 0)
            {
                _logger.LogWarning("Command exited with non-zero code: {ExitCode}", process.ExitCode);
                context.AddWarning($"Command exited with non-zero code: {process.ExitCode}");
                context.AddLog(LogLevelConverter.ToTarsLogLevel(Microsoft.Extensions.Logging.LogLevel.Warning), $"Command error output: {error}");
            }

            return string.IsNullOrEmpty(error) ? output : $"{output}\n{error}";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing command: {Command}", command);

            // Update context with error
            var context = GetExecutionContext(contextId);
            context?.AddError($"Error executing command: {command}", "SafeExecutionEnvironment", ex);

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
        return _virtualFileSystem.GetModifiedFiles(contextId);
    }

    /// <summary>
    /// Gets the list of created files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The list of created files</returns>
    public List<string> GetCreatedFiles(string contextId)
    {
        return _virtualFileSystem.GetCreatedFiles(contextId);
    }

    /// <summary>
    /// Gets the list of deleted files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The list of deleted files</returns>
    public List<string> GetDeletedFiles(string contextId)
    {
        return _virtualFileSystem.GetDeletedFiles(contextId);
    }

    /// <summary>
    /// Gets the list of backed up files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The list of backed up files</returns>
    public List<string> GetBackedUpFiles(string contextId)
    {
        return _virtualFileSystem.GetBackedUpFiles(contextId);
    }

    /// <summary>
    /// Gets the available execution environment options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        var options = new Dictionary<string, string>
        {
            { "WorkingDirectory", "The working directory for the execution environment" },
            { "BackupDirectory", "The backup directory for the execution environment" },
            { "OutputDirectory", "The output directory for the execution environment" },
            { "TimeoutMs", "The execution timeout in milliseconds" }
        };

        // Add permission options
        var permissionOptions = _permissionManager.GetAvailableOptions();
        foreach (var option in permissionOptions)
        {
            options[option.Key] = option.Value;
        }

        // Add file system options
        var fileSystemOptions = _virtualFileSystem.GetAvailableOptions();
        foreach (var option in fileSystemOptions)
        {
            options[option.Key] = option.Value;
        }

        return options;
    }
}
