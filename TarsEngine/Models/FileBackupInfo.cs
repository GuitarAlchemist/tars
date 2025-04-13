namespace TarsEngine.Models;

/// <summary>
/// Represents information about a file backup
/// </summary>
public class FileBackupInfo
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the backup file path
    /// </summary>
    public string BackupFilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the backup time
    /// </summary>
    public DateTime BackupTime { get; set; }

    /// <summary>
    /// Gets or sets the file hash
    /// </summary>
    public string FileHash { get; set; } = string.Empty;
}
