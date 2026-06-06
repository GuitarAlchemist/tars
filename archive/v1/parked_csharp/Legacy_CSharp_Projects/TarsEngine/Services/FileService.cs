using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Implementation of the file service
/// </summary>
public class FileService : IFileService
{
    private readonly ILogger<FileService> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="FileService"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public FileService(ILogger<FileService> logger)
    {
        _logger = logger;
    }

    /// <inheritdoc/>
    public async Task<string> ReadFileAsync(string filePath)
    {
        try
        {
            _logger.LogInformation($"Reading file: {filePath}");
            return await File.ReadAllTextAsync(filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error reading file: {filePath}");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> WriteFileAsync(string filePath, string content)
    {
        try
        {
            _logger.LogInformation($"Writing to file: {filePath}");
            await File.WriteAllTextAsync(filePath, content);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error writing to file: {filePath}");
            return false;
        }
    }

    /// <inheritdoc/>
    public Task<List<string>> GetFilesAsync(string directoryPath, string pattern)
    {
        try
        {
            _logger.LogInformation($"Getting files in {directoryPath} with pattern {pattern}");
            
            var patterns = pattern.Split(';');
            var files = new List<string>();
            
            foreach (var p in patterns)
            {
                files.AddRange(Directory.GetFiles(directoryPath, p, SearchOption.AllDirectories));
            }
            
            return Task.FromResult(files);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting files in {directoryPath} with pattern {pattern}");
            return Task.FromResult(new List<string>());
        }
    }

    /// <inheritdoc/>
    public Task<bool> FileExistsAsync(string filePath)
    {
        try
        {
            _logger.LogInformation($"Checking if file exists: {filePath}");
            return Task.FromResult(File.Exists(filePath));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error checking if file exists: {filePath}");
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> CreateDirectoryAsync(string directoryPath)
    {
        try
        {
            _logger.LogInformation($"Creating directory: {directoryPath}");
            Directory.CreateDirectory(directoryPath);
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error creating directory: {directoryPath}");
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> DeleteFileAsync(string filePath)
    {
        try
        {
            _logger.LogInformation($"Deleting file: {filePath}");
            File.Delete(filePath);
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error deleting file: {filePath}");
            return Task.FromResult(false);
        }
    }
}
