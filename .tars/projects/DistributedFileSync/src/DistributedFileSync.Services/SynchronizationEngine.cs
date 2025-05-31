using DistributedFileSync.Core.Interfaces;
using DistributedFileSync.Core.Models;
using Microsoft.Extensions.Logging;
using System.Security.Cryptography;
using System.Text;

namespace DistributedFileSync.Services;

/// <summary>
/// Core synchronization engine implementation
/// Implemented by: Senior Developer Agent (Bob)
/// Performance optimized by: Performance Engineer Agent (Dave)
/// Security hardened by: Security Specialist Agent (Eve)
/// </summary>
public class SynchronizationEngine : ISynchronizationEngine
{
    private readonly ILogger<SynchronizationEngine> _logger;
    private readonly IFileWatcherService _fileWatcher;
    private readonly INodeDiscoveryService _nodeDiscovery;
    private readonly IConflictResolutionService _conflictResolver;
    private readonly Dictionary<string, FileMetadata> _fileMetadataCache;
    private readonly SemaphoreSlim _syncSemaphore;
    private readonly CancellationTokenSource _cancellationTokenSource;
    private bool _isRunning;

    public event EventHandler<FileSyncEventArgs>? SyncStarted;
    public event EventHandler<FileSyncEventArgs>? SyncCompleted;
    public event EventHandler<FileSyncErrorEventArgs>? SyncFailed;
    public event EventHandler<ConflictDetectedEventArgs>? ConflictDetected;

    public SynchronizationEngine(
        ILogger<SynchronizationEngine> logger,
        IFileWatcherService fileWatcher,
        INodeDiscoveryService nodeDiscovery,
        IConflictResolutionService conflictResolver)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _fileWatcher = fileWatcher ?? throw new ArgumentNullException(nameof(fileWatcher));
        _nodeDiscovery = nodeDiscovery ?? throw new ArgumentNullException(nameof(nodeDiscovery));
        _conflictResolver = conflictResolver ?? throw new ArgumentNullException(nameof(conflictResolver));
        
        _fileMetadataCache = new Dictionary<string, FileMetadata>();
        _syncSemaphore = new SemaphoreSlim(10, 10); // Max 10 concurrent syncs (Performance optimization)
        _cancellationTokenSource = new CancellationTokenSource();
    }

    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        if (_isRunning)
        {
            _logger.LogWarning("Synchronization engine is already running");
            return;
        }

        _logger.LogInformation("Starting synchronization engine");

        try
        {
            // Start file watcher
            await _fileWatcher.StartAsync(cancellationToken);
            
            // Subscribe to file change events
            _fileWatcher.FileChanged += OnFileChanged;
            _fileWatcher.FileCreated += OnFileCreated;
            _fileWatcher.FileDeleted += OnFileDeleted;

            _isRunning = true;
            _logger.LogInformation("Synchronization engine started successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start synchronization engine");
            throw;
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (!_isRunning)
        {
            _logger.LogWarning("Synchronization engine is not running");
            return;
        }

        _logger.LogInformation("Stopping synchronization engine");

        try
        {
            // Cancel all ongoing operations
            _cancellationTokenSource.Cancel();

            // Stop file watcher
            await _fileWatcher.StopAsync(cancellationToken);

            // Unsubscribe from events
            _fileWatcher.FileChanged -= OnFileChanged;
            _fileWatcher.FileCreated -= OnFileCreated;
            _fileWatcher.FileDeleted -= OnFileDeleted;

            _isRunning = false;
            _logger.LogInformation("Synchronization engine stopped successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping synchronization engine");
            throw;
        }
    }

    public async Task<SyncResult> SynchronizeFileAsync(string filePath, IEnumerable<SyncNode> targetNodes, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(filePath))
        {
            return new SyncResult
            {
                Success = false,
                ErrorMessage = $"File not found: {filePath}"
            };
        }

        var startTime = DateTime.UtcNow;
        var result = new SyncResult();

        try
        {
            await _syncSemaphore.WaitAsync(cancellationToken);

            _logger.LogInformation("Starting synchronization of file: {FilePath}", filePath);

            // Generate file metadata
            var metadata = await GenerateFileMetadataAsync(filePath);
            
            // Check for conflicts
            var conflicts = await DetectConflictsAsync(metadata, targetNodes);
            if (conflicts.Any())
            {
                result.Conflicts.AddRange(conflicts);
                OnConflictDetected(new ConflictDetectedEventArgs { Conflict = conflicts.First() });
                
                // Auto-resolve if possible
                foreach (var conflict in conflicts)
                {
                    var resolution = await ResolveConflictAsync(conflict.FileId, ConflictResolutionStrategy.ThreeWayMerge, cancellationToken);
                    if (!resolution.Resolved)
                    {
                        result.Success = false;
                        result.ErrorMessage = $"Failed to resolve conflict: {resolution.ErrorMessage}";
                        return result;
                    }
                }
            }

            // Notify sync started
            OnSyncStarted(new FileSyncEventArgs { FileMetadata = metadata });

            // Synchronize with each target node
            var syncTasks = targetNodes.Select(node => SynchronizeWithNodeAsync(metadata, node, cancellationToken));
            var syncResults = await Task.WhenAll(syncTasks);

            // Aggregate results
            result.Success = syncResults.All(r => r.Success);
            result.FilesSynchronized = syncResults.Count(r => r.Success);
            result.BytesTransferred = syncResults.Sum(r => r.BytesTransferred);
            result.SynchronizedNodes.AddRange(syncResults.Where(r => r.Success).SelectMany(r => r.SynchronizedNodes));

            if (!result.Success)
            {
                var errors = syncResults.Where(r => !r.Success).Select(r => r.ErrorMessage);
                result.ErrorMessage = string.Join("; ", errors);
            }

            result.Duration = DateTime.UtcNow - startTime;

            // Update cache
            _fileMetadataCache[filePath] = metadata;

            // Notify completion
            OnSyncCompleted(new FileSyncEventArgs { FileMetadata = metadata });

            _logger.LogInformation("File synchronization completed: {FilePath}, Success: {Success}", filePath, result.Success);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error synchronizing file: {FilePath}", filePath);
            result.Success = false;
            result.ErrorMessage = ex.Message;
            
            OnSyncFailed(new FileSyncErrorEventArgs 
            { 
                FileMetadata = new FileMetadata { FilePath = filePath },
                ErrorMessage = ex.Message,
                Exception = ex
            });
        }
        finally
        {
            _syncSemaphore.Release();
        }

        return result;
    }

    public async Task<SyncResult> SynchronizeDirectoryAsync(string directoryPath, IEnumerable<SyncNode> targetNodes, bool recursive = true, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(directoryPath))
        {
            return new SyncResult
            {
                Success = false,
                ErrorMessage = $"Directory not found: {directoryPath}"
            };
        }

        _logger.LogInformation("Starting directory synchronization: {DirectoryPath}, Recursive: {Recursive}", directoryPath, recursive);

        var overallResult = new SyncResult { Success = true };
        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        try
        {
            var files = Directory.GetFiles(directoryPath, "*", searchOption);
            var syncTasks = files.Select(file => SynchronizeFileAsync(file, targetNodes, cancellationToken));
            var results = await Task.WhenAll(syncTasks);

            // Aggregate results
            overallResult.Success = results.All(r => r.Success);
            overallResult.FilesSynchronized = results.Sum(r => r.FilesSynchronized);
            overallResult.BytesTransferred = results.Sum(r => r.BytesTransferred);
            overallResult.Duration = results.Max(r => r.Duration);
            
            foreach (var result in results)
            {
                overallResult.SynchronizedNodes.AddRange(result.SynchronizedNodes);
                overallResult.Conflicts.AddRange(result.Conflicts);
            }

            if (!overallResult.Success)
            {
                var errors = results.Where(r => !r.Success).Select(r => r.ErrorMessage);
                overallResult.ErrorMessage = string.Join("; ", errors);
            }

            _logger.LogInformation("Directory synchronization completed: {DirectoryPath}, Files: {FileCount}, Success: {Success}", 
                directoryPath, overallResult.FilesSynchronized, overallResult.Success);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error synchronizing directory: {DirectoryPath}", directoryPath);
            overallResult.Success = false;
            overallResult.ErrorMessage = ex.Message;
        }

        return overallResult;
    }

    public async Task<FileMetadata?> GetFileSyncStatusAsync(string filePath)
    {
        if (_fileMetadataCache.TryGetValue(filePath, out var cachedMetadata))
        {
            return cachedMetadata;
        }

        if (File.Exists(filePath))
        {
            var metadata = await GenerateFileMetadataAsync(filePath);
            _fileMetadataCache[filePath] = metadata;
            return metadata;
        }

        return null;
    }

    public async Task<IEnumerable<FileMetadata>> GetActiveSynchronizationsAsync()
    {
        return await Task.FromResult(_fileMetadataCache.Values.Where(m => m.IsSyncing));
    }

    public async Task<ConflictResolutionResult> ResolveConflictAsync(Guid fileId, ConflictResolutionStrategy strategy, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Resolving conflict for file: {FileId}, Strategy: {Strategy}", fileId, strategy);

        try
        {
            return await _conflictResolver.ResolveConflictAsync(fileId, strategy, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error resolving conflict for file: {FileId}", fileId);
            return new ConflictResolutionResult
            {
                Resolved = false,
                ErrorMessage = ex.Message
            };
        }
    }

    private async Task<FileMetadata> GenerateFileMetadataAsync(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        var contentHash = await CalculateFileHashAsync(filePath);

        return new FileMetadata
        {
            FilePath = filePath,
            FileName = fileInfo.Name,
            FileSize = fileInfo.Length,
            ContentHash = contentHash,
            LastModified = fileInfo.LastWriteTimeUtc,
            CreatedAt = fileInfo.CreationTimeUtc,
            Status = SyncStatus.Pending,
            MetadataUpdated = DateTime.UtcNow
        };
    }

    private async Task<string> CalculateFileHashAsync(string filePath)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(filePath);
        var hashBytes = await sha256.ComputeHashAsync(stream);
        return Convert.ToBase64String(hashBytes);
    }

    private async Task<List<FileConflict>> DetectConflictsAsync(FileMetadata metadata, IEnumerable<SyncNode> targetNodes)
    {
        var conflicts = new List<FileConflict>();
        
        // Implementation would check for conflicts with target nodes
        // This is a simplified version
        await Task.Delay(10); // Simulate async operation
        
        return conflicts;
    }

    private async Task<SyncResult> SynchronizeWithNodeAsync(FileMetadata metadata, SyncNode node, CancellationToken cancellationToken)
    {
        // Implementation would use gRPC to synchronize with the target node
        // This is a simplified version
        await Task.Delay(100, cancellationToken); // Simulate network operation
        
        return new SyncResult
        {
            Success = true,
            FilesSynchronized = 1,
            BytesTransferred = metadata.FileSize,
            SynchronizedNodes = new List<SyncNode> { node }
        };
    }

    private void OnFileChanged(object? sender, FileSystemEventArgs e)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                var nodes = await _nodeDiscovery.GetAvailableNodesAsync();
                await SynchronizeFileAsync(e.FullPath, nodes);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling file change: {FilePath}", e.FullPath);
            }
        });
    }

    private void OnFileCreated(object? sender, FileSystemEventArgs e)
    {
        OnFileChanged(sender, e);
    }

    private void OnFileDeleted(object? sender, FileSystemEventArgs e)
    {
        _fileMetadataCache.Remove(e.FullPath);
        _logger.LogInformation("File deleted: {FilePath}", e.FullPath);
    }

    private void OnSyncStarted(FileSyncEventArgs args) => SyncStarted?.Invoke(this, args);
    private void OnSyncCompleted(FileSyncEventArgs args) => SyncCompleted?.Invoke(this, args);
    private void OnSyncFailed(FileSyncErrorEventArgs args) => SyncFailed?.Invoke(this, args);
    private void OnConflictDetected(ConflictDetectedEventArgs args) => ConflictDetected?.Invoke(this, args);
}

// Supporting service interfaces (would be implemented in separate files)
public interface IFileWatcherService
{
    Task StartAsync(CancellationToken cancellationToken);
    Task StopAsync(CancellationToken cancellationToken);
    event EventHandler<FileSystemEventArgs> FileChanged;
    event EventHandler<FileSystemEventArgs> FileCreated;
    event EventHandler<FileSystemEventArgs> FileDeleted;
}

public interface INodeDiscoveryService
{
    Task<IEnumerable<SyncNode>> GetAvailableNodesAsync();
    Task RegisterNodeAsync(SyncNode node);
}

public interface IConflictResolutionService
{
    Task<ConflictResolutionResult> ResolveConflictAsync(Guid fileId, ConflictResolutionStrategy strategy, CancellationToken cancellationToken);
}
