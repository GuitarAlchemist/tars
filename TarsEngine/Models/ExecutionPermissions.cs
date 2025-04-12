namespace TarsEngine.Models;

/// <summary>
/// Represents the permissions for execution
/// </summary>
public class ExecutionPermissions
{
    /// <summary>
    /// Gets or sets whether file system access is allowed
    /// </summary>
    public bool AllowFileSystemAccess { get; set; } = true;

    /// <summary>
    /// Gets or sets whether network access is allowed
    /// </summary>
    public bool AllowNetworkAccess { get; set; } = false;

    /// <summary>
    /// Gets or sets whether process execution is allowed
    /// </summary>
    public bool AllowProcessExecution { get; set; } = false;

    /// <summary>
    /// Gets or sets whether registry access is allowed
    /// </summary>
    public bool AllowRegistryAccess { get; set; } = false;

    /// <summary>
    /// Gets or sets whether environment variable access is allowed
    /// </summary>
    public bool AllowEnvironmentAccess { get; set; } = true;

    /// <summary>
    /// Gets or sets whether file creation is allowed
    /// </summary>
    public bool AllowFileCreation { get; set; } = true;

    /// <summary>
    /// Gets or sets whether file modification is allowed
    /// </summary>
    public bool AllowFileModification { get; set; } = true;

    /// <summary>
    /// Gets or sets whether file deletion is allowed
    /// </summary>
    public bool AllowFileDeletion { get; set; } = true;

    /// <summary>
    /// Gets or sets the list of allowed file paths
    /// </summary>
    public List<string> AllowedFilePaths { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of denied file paths
    /// </summary>
    public List<string> DeniedFilePaths { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of allowed file extensions
    /// </summary>
    public List<string> AllowedFileExtensions { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of denied file extensions
    /// </summary>
    public List<string> DeniedFileExtensions { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of allowed process names
    /// </summary>
    public List<string> AllowedProcessNames { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of denied process names
    /// </summary>
    public List<string> DeniedProcessNames { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of allowed network hosts
    /// </summary>
    public List<string> AllowedNetworkHosts { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of denied network hosts
    /// </summary>
    public List<string> DeniedNetworkHosts { get; set; } = new();

    /// <summary>
    /// Gets or sets the maximum file size in bytes
    /// </summary>
    public long MaxFileSize { get; set; } = 10 * 1024 * 1024; // 10 MB

    /// <summary>
    /// Gets or sets the maximum number of files
    /// </summary>
    public int MaxFileCount { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum execution time in milliseconds
    /// </summary>
    public int MaxExecutionTimeMs { get; set; } = 60000; // 1 minute

    /// <summary>
    /// Gets or sets the maximum memory usage in bytes
    /// </summary>
    public long MaxMemoryUsage { get; set; } = 100 * 1024 * 1024; // 100 MB

    /// <summary>
    /// Checks if a file path is allowed
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the file path is allowed, false otherwise</returns>
    public bool IsFilePathAllowed(string filePath)
    {
        if (!AllowFileSystemAccess)
        {
            return false;
        }

        // Check denied paths
        foreach (var deniedPath in DeniedFilePaths)
        {
            if (filePath.StartsWith(deniedPath, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        // Check allowed paths
        if (AllowedFilePaths.Count > 0)
        {
            bool allowed = false;
            foreach (var allowedPath in AllowedFilePaths)
            {
                if (filePath.StartsWith(allowedPath, StringComparison.OrdinalIgnoreCase))
                {
                    allowed = true;
                    break;
                }
            }
            if (!allowed)
            {
                return false;
            }
        }

        // Check denied extensions
        var extension = Path.GetExtension(filePath);
        foreach (var deniedExtension in DeniedFileExtensions)
        {
            if (extension.Equals(deniedExtension, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        // Check allowed extensions
        if (AllowedFileExtensions.Count > 0)
        {
            bool allowed = false;
            foreach (var allowedExtension in AllowedFileExtensions)
            {
                if (extension.Equals(allowedExtension, StringComparison.OrdinalIgnoreCase))
                {
                    allowed = true;
                    break;
                }
            }
            if (!allowed)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Checks if a process name is allowed
    /// </summary>
    /// <param name="processName">The process name</param>
    /// <returns>True if the process name is allowed, false otherwise</returns>
    public bool IsProcessNameAllowed(string processName)
    {
        if (!AllowProcessExecution)
        {
            return false;
        }

        // Check denied process names
        foreach (var deniedName in DeniedProcessNames)
        {
            if (processName.Equals(deniedName, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        // Check allowed process names
        if (AllowedProcessNames.Count > 0)
        {
            bool allowed = false;
            foreach (var allowedName in AllowedProcessNames)
            {
                if (processName.Equals(allowedName, StringComparison.OrdinalIgnoreCase))
                {
                    allowed = true;
                    break;
                }
            }
            if (!allowed)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Checks if a network host is allowed
    /// </summary>
    /// <param name="host">The network host</param>
    /// <returns>True if the network host is allowed, false otherwise</returns>
    public bool IsNetworkHostAllowed(string host)
    {
        if (!AllowNetworkAccess)
        {
            return false;
        }

        // Check denied hosts
        foreach (var deniedHost in DeniedNetworkHosts)
        {
            if (host.Equals(deniedHost, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        // Check allowed hosts
        if (AllowedNetworkHosts.Count > 0)
        {
            bool allowed = false;
            foreach (var allowedHost in AllowedNetworkHosts)
            {
                if (host.Equals(allowedHost, StringComparison.OrdinalIgnoreCase))
                {
                    allowed = true;
                    break;
                }
            }
            if (!allowed)
            {
                return false;
            }
        }

        return true;
    }
}
