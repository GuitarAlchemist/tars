using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Manages security permissions for execution
/// </summary>
public class PermissionManager
{
    private readonly ILogger<PermissionManager> _logger;
    private readonly Dictionary<string, ExecutionPermissions> _contextPermissions = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="PermissionManager"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public PermissionManager(ILogger<PermissionManager> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Creates permissions for an execution context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="mode">The execution mode</param>
    /// <param name="environment">The execution environment</param>
    /// <param name="options">Optional permission options</param>
    /// <returns>The execution permissions</returns>
    public ExecutionPermissions CreatePermissions(
        string contextId,
        ExecutionMode mode,
        ExecutionEnvironment environment,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Creating permissions for execution context: {ContextId}", contextId);

            var permissions = new ExecutionPermissions();

            // Set permissions based on execution mode
            switch (mode)
            {
                case ExecutionMode.DryRun:
                    // Dry run mode has read-only permissions
                    permissions.AllowFileSystemAccess = true;
                    permissions.AllowNetworkAccess = false;
                    permissions.AllowProcessExecution = false;
                    permissions.AllowRegistryAccess = false;
                    permissions.AllowEnvironmentAccess = true;
                    permissions.AllowFileCreation = false;
                    permissions.AllowFileModification = false;
                    permissions.AllowFileDeletion = false;
                    break;

                case ExecutionMode.Real:
                    // Real mode has full permissions (within environment constraints)
                    permissions.AllowFileSystemAccess = true;
                    permissions.AllowNetworkAccess = environment != ExecutionEnvironment.Sandbox;
                    permissions.AllowProcessExecution = environment != ExecutionEnvironment.Sandbox;
                    permissions.AllowRegistryAccess = environment != ExecutionEnvironment.Sandbox;
                    permissions.AllowEnvironmentAccess = true;
                    permissions.AllowFileCreation = true;
                    permissions.AllowFileModification = true;
                    permissions.AllowFileDeletion = true;
                    break;

                case ExecutionMode.Interactive:
                    // Interactive mode has full permissions but requires confirmation
                    permissions.AllowFileSystemAccess = true;
                    permissions.AllowNetworkAccess = environment != ExecutionEnvironment.Sandbox;
                    permissions.AllowProcessExecution = environment != ExecutionEnvironment.Sandbox;
                    permissions.AllowRegistryAccess = environment != ExecutionEnvironment.Sandbox;
                    permissions.AllowEnvironmentAccess = true;
                    permissions.AllowFileCreation = true;
                    permissions.AllowFileModification = true;
                    permissions.AllowFileDeletion = true;
                    break;
            }

            // Set permissions based on execution environment
            switch (environment)
            {
                case ExecutionEnvironment.Sandbox:
                    // Sandbox environment has restricted permissions
                    permissions.AllowNetworkAccess = false;
                    permissions.AllowProcessExecution = false;
                    permissions.AllowRegistryAccess = false;
                    permissions.MaxFileSize = 5 * 1024 * 1024; // 5 MB
                    permissions.MaxFileCount = 50;
                    permissions.MaxExecutionTimeMs = 30000; // 30 seconds
                    permissions.MaxMemoryUsage = 50 * 1024 * 1024; // 50 MB
                    break;

                case ExecutionEnvironment.Development:
                    // Development environment has relaxed permissions
                    permissions.MaxFileSize = 20 * 1024 * 1024; // 20 MB
                    permissions.MaxFileCount = 200;
                    permissions.MaxExecutionTimeMs = 120000; // 2 minutes
                    permissions.MaxMemoryUsage = 200 * 1024 * 1024; // 200 MB
                    break;

                case ExecutionEnvironment.Testing:
                case ExecutionEnvironment.Staging:
                case ExecutionEnvironment.Production:
                    // Other environments have standard permissions
                    permissions.MaxFileSize = 10 * 1024 * 1024; // 10 MB
                    permissions.MaxFileCount = 100;
                    permissions.MaxExecutionTimeMs = 60000; // 1 minute
                    permissions.MaxMemoryUsage = 100 * 1024 * 1024; // 100 MB
                    break;
            }

            // Apply custom options
            if (options != null)
            {
                // File system access
                if (options.TryGetValue("AllowFileSystemAccess", out var allowFileSystemAccess))
                {
                    permissions.AllowFileSystemAccess = bool.Parse(allowFileSystemAccess);
                }

                // Network access
                if (options.TryGetValue("AllowNetworkAccess", out var allowNetworkAccess))
                {
                    permissions.AllowNetworkAccess = bool.Parse(allowNetworkAccess);
                }

                // Process execution
                if (options.TryGetValue("AllowProcessExecution", out var allowProcessExecution))
                {
                    permissions.AllowProcessExecution = bool.Parse(allowProcessExecution);
                }

                // Registry access
                if (options.TryGetValue("AllowRegistryAccess", out var allowRegistryAccess))
                {
                    permissions.AllowRegistryAccess = bool.Parse(allowRegistryAccess);
                }

                // Environment access
                if (options.TryGetValue("AllowEnvironmentAccess", out var allowEnvironmentAccess))
                {
                    permissions.AllowEnvironmentAccess = bool.Parse(allowEnvironmentAccess);
                }

                // File creation
                if (options.TryGetValue("AllowFileCreation", out var allowFileCreation))
                {
                    permissions.AllowFileCreation = bool.Parse(allowFileCreation);
                }

                // File modification
                if (options.TryGetValue("AllowFileModification", out var allowFileModification))
                {
                    permissions.AllowFileModification = bool.Parse(allowFileModification);
                }

                // File deletion
                if (options.TryGetValue("AllowFileDeletion", out var allowFileDeletion))
                {
                    permissions.AllowFileDeletion = bool.Parse(allowFileDeletion);
                }

                // Allowed file paths
                if (options.TryGetValue("AllowedFilePaths", out var allowedFilePaths))
                {
                    permissions.AllowedFilePaths = allowedFilePaths.Split(';').ToList();
                }

                // Denied file paths
                if (options.TryGetValue("DeniedFilePaths", out var deniedFilePaths))
                {
                    permissions.DeniedFilePaths = deniedFilePaths.Split(';').ToList();
                }

                // Allowed file extensions
                if (options.TryGetValue("AllowedFileExtensions", out var allowedFileExtensions))
                {
                    permissions.AllowedFileExtensions = allowedFileExtensions.Split(';').ToList();
                }

                // Denied file extensions
                if (options.TryGetValue("DeniedFileExtensions", out var deniedFileExtensions))
                {
                    permissions.DeniedFileExtensions = deniedFileExtensions.Split(';').ToList();
                }

                // Allowed process names
                if (options.TryGetValue("AllowedProcessNames", out var allowedProcessNames))
                {
                    permissions.AllowedProcessNames = allowedProcessNames.Split(';').ToList();
                }

                // Denied process names
                if (options.TryGetValue("DeniedProcessNames", out var deniedProcessNames))
                {
                    permissions.DeniedProcessNames = deniedProcessNames.Split(';').ToList();
                }

                // Allowed network hosts
                if (options.TryGetValue("AllowedNetworkHosts", out var allowedNetworkHosts))
                {
                    permissions.AllowedNetworkHosts = allowedNetworkHosts.Split(';').ToList();
                }

                // Denied network hosts
                if (options.TryGetValue("DeniedNetworkHosts", out var deniedNetworkHosts))
                {
                    permissions.DeniedNetworkHosts = deniedNetworkHosts.Split(';').ToList();
                }

                // Max file size
                if (options.TryGetValue("MaxFileSize", out var maxFileSize))
                {
                    permissions.MaxFileSize = long.Parse(maxFileSize);
                }

                // Max file count
                if (options.TryGetValue("MaxFileCount", out var maxFileCount))
                {
                    permissions.MaxFileCount = int.Parse(maxFileCount);
                }

                // Max execution time
                if (options.TryGetValue("MaxExecutionTimeMs", out var maxExecutionTimeMs))
                {
                    permissions.MaxExecutionTimeMs = int.Parse(maxExecutionTimeMs);
                }

                // Max memory usage
                if (options.TryGetValue("MaxMemoryUsage", out var maxMemoryUsage))
                {
                    permissions.MaxMemoryUsage = long.Parse(maxMemoryUsage);
                }
            }

            // Store permissions for the context
            _contextPermissions[contextId] = permissions;

            _logger.LogInformation("Created permissions for execution context: {ContextId}", contextId);
            return permissions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating permissions for execution context: {ContextId}", contextId);
            throw;
        }
    }

    /// <summary>
    /// Gets permissions for an execution context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The execution permissions, or null if not found</returns>
    public ExecutionPermissions? GetPermissions(string contextId)
    {
        if (_contextPermissions.TryGetValue(contextId, out var permissions))
        {
            return permissions;
        }

        _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
        return null;
    }

    /// <summary>
    /// Updates permissions for an execution context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="permissions">The execution permissions</param>
    public void UpdatePermissions(string contextId, ExecutionPermissions permissions)
    {
        _contextPermissions[contextId] = permissions;
        _logger.LogInformation("Updated permissions for execution context: {ContextId}", contextId);
    }

    /// <summary>
    /// Removes permissions for an execution context
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    public void RemovePermissions(string contextId)
    {
        if (_contextPermissions.Remove(contextId))
        {
            _logger.LogInformation("Removed permissions for execution context: {ContextId}", contextId);
        }
        else
        {
            _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
        }
    }

    /// <summary>
    /// Validates a file operation
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="filePath">The file path</param>
    /// <param name="operation">The file operation</param>
    /// <returns>True if the operation is allowed, false otherwise</returns>
    public bool ValidateFileOperation(string contextId, string filePath, FileOperation operation)
    {
        try
        {
            var permissions = GetPermissions(contextId);
            if (permissions == null)
            {
                _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
                return false;
            }

            // Check file system access
            if (!permissions.AllowFileSystemAccess)
            {
                _logger.LogWarning("File system access not allowed for execution context: {ContextId}", contextId);
                return false;
            }

            // Check file path
            if (!permissions.IsFilePathAllowed(filePath))
            {
                _logger.LogWarning("File path not allowed: {FilePath}", filePath);
                return false;
            }

            // Check operation-specific permissions
            switch (operation)
            {
                case FileOperation.Read:
                    // Reading is always allowed if file system access is allowed
                    return true;

                case FileOperation.Create:
                    if (!permissions.AllowFileCreation)
                    {
                        _logger.LogWarning("File creation not allowed for execution context: {ContextId}", contextId);
                        return false;
                    }
                    break;

                case FileOperation.Modify:
                    if (!permissions.AllowFileModification)
                    {
                        _logger.LogWarning("File modification not allowed for execution context: {ContextId}", contextId);
                        return false;
                    }
                    break;

                case FileOperation.Delete:
                    if (!permissions.AllowFileDeletion)
                    {
                        _logger.LogWarning("File deletion not allowed for execution context: {ContextId}", contextId);
                        return false;
                    }
                    break;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating file operation for execution context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Validates a process execution
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="processName">The process name</param>
    /// <returns>True if the operation is allowed, false otherwise</returns>
    public bool ValidateProcessExecution(string contextId, string processName)
    {
        try
        {
            var permissions = GetPermissions(contextId);
            if (permissions == null)
            {
                _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
                return false;
            }

            // Check process execution
            if (!permissions.AllowProcessExecution)
            {
                _logger.LogWarning("Process execution not allowed for execution context: {ContextId}", contextId);
                return false;
            }

            // Check process name
            if (!permissions.IsProcessNameAllowed(processName))
            {
                _logger.LogWarning("Process name not allowed: {ProcessName}", processName);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating process execution for execution context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Validates a network operation
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="host">The network host</param>
    /// <returns>True if the operation is allowed, false otherwise</returns>
    public bool ValidateNetworkOperation(string contextId, string host)
    {
        try
        {
            var permissions = GetPermissions(contextId);
            if (permissions == null)
            {
                _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
                return false;
            }

            // Check network access
            if (!permissions.AllowNetworkAccess)
            {
                _logger.LogWarning("Network access not allowed for execution context: {ContextId}", contextId);
                return false;
            }

            // Check host
            if (!permissions.IsNetworkHostAllowed(host))
            {
                _logger.LogWarning("Network host not allowed: {Host}", host);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating network operation for execution context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Validates a registry operation
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>True if the operation is allowed, false otherwise</returns>
    public bool ValidateRegistryOperation(string contextId)
    {
        try
        {
            var permissions = GetPermissions(contextId);
            if (permissions == null)
            {
                _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
                return false;
            }

            // Check registry access
            if (!permissions.AllowRegistryAccess)
            {
                _logger.LogWarning("Registry access not allowed for execution context: {ContextId}", contextId);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating registry operation for execution context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Validates an environment operation
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>True if the operation is allowed, false otherwise</returns>
    public bool ValidateEnvironmentOperation(string contextId)
    {
        try
        {
            var permissions = GetPermissions(contextId);
            if (permissions == null)
            {
                _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
                return false;
            }

            // Check environment access
            if (!permissions.AllowEnvironmentAccess)
            {
                _logger.LogWarning("Environment access not allowed for execution context: {ContextId}", contextId);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating environment operation for execution context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Enforces file size limit
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="fileSize">The file size</param>
    /// <returns>True if the file size is within limits, false otherwise</returns>
    public bool EnforceFileSizeLimit(string contextId, long fileSize)
    {
        try
        {
            var permissions = GetPermissions(contextId);
            if (permissions == null)
            {
                _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
                return false;
            }

            if (fileSize > permissions.MaxFileSize)
            {
                _logger.LogWarning("File size exceeds limit: {FileSize} > {MaxFileSize}", fileSize, permissions.MaxFileSize);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enforcing file size limit for execution context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Enforces file count limit
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="fileCount">The file count</param>
    /// <returns>True if the file count is within limits, false otherwise</returns>
    public bool EnforceFileCountLimit(string contextId, int fileCount)
    {
        try
        {
            var permissions = GetPermissions(contextId);
            if (permissions == null)
            {
                _logger.LogWarning("Permissions not found for execution context: {ContextId}", contextId);
                return false;
            }

            if (fileCount > permissions.MaxFileCount)
            {
                _logger.LogWarning("File count exceeds limit: {FileCount} > {MaxFileCount}", fileCount, permissions.MaxFileCount);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enforcing file count limit for execution context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Gets the available permission options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "AllowFileSystemAccess", "Whether to allow file system access (default: true)" },
            { "AllowNetworkAccess", "Whether to allow network access (default: depends on environment)" },
            { "AllowProcessExecution", "Whether to allow process execution (default: depends on environment)" },
            { "AllowRegistryAccess", "Whether to allow registry access (default: depends on environment)" },
            { "AllowEnvironmentAccess", "Whether to allow environment access (default: true)" },
            { "AllowFileCreation", "Whether to allow file creation (default: depends on mode)" },
            { "AllowFileModification", "Whether to allow file modification (default: depends on mode)" },
            { "AllowFileDeletion", "Whether to allow file deletion (default: depends on mode)" },
            { "AllowedFilePaths", "Semicolon-separated list of allowed file paths" },
            { "DeniedFilePaths", "Semicolon-separated list of denied file paths" },
            { "AllowedFileExtensions", "Semicolon-separated list of allowed file extensions" },
            { "DeniedFileExtensions", "Semicolon-separated list of denied file extensions" },
            { "AllowedProcessNames", "Semicolon-separated list of allowed process names" },
            { "DeniedProcessNames", "Semicolon-separated list of denied process names" },
            { "AllowedNetworkHosts", "Semicolon-separated list of allowed network hosts" },
            { "DeniedNetworkHosts", "Semicolon-separated list of denied network hosts" },
            { "MaxFileSize", "Maximum file size in bytes" },
            { "MaxFileCount", "Maximum number of files" },
            { "MaxExecutionTimeMs", "Maximum execution time in milliseconds" },
            { "MaxMemoryUsage", "Maximum memory usage in bytes" }
        };
    }
}
