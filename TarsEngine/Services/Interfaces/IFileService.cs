using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for file service
/// </summary>
public interface IFileService
{
    /// <summary>
    /// Reads a file asynchronously
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>File content</returns>
    Task<string> ReadFileAsync(string filePath);
    
    /// <summary>
    /// Writes to a file asynchronously
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="content">Content to write</param>
    /// <returns>True if successful</returns>
    Task<bool> WriteFileAsync(string filePath, string content);
    
    /// <summary>
    /// Gets files matching a pattern
    /// </summary>
    /// <param name="directoryPath">Path to the directory</param>
    /// <param name="pattern">File pattern (e.g., "*.cs;*.fs")</param>
    /// <returns>List of file paths</returns>
    Task<List<string>> GetFilesAsync(string directoryPath, string pattern);
    
    /// <summary>
    /// Checks if a file exists
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>True if the file exists</returns>
    Task<bool> FileExistsAsync(string filePath);
    
    /// <summary>
    /// Creates a directory
    /// </summary>
    /// <param name="directoryPath">Path to the directory</param>
    /// <returns>True if successful</returns>
    Task<bool> CreateDirectoryAsync(string directoryPath);
    
    /// <summary>
    /// Deletes a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>True if successful</returns>
    Task<bool> DeleteFileAsync(string filePath);
}
