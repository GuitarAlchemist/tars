using DistributedFileSync.Core.Interfaces;
using DistributedFileSync.Core.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace DistributedFileSync.Api.Controllers;

/// <summary>
/// File synchronization API controller
/// Designed by: Architect Agent (Alice)
/// Implemented by: Senior Developer Agent (Bob)
/// Security reviewed by: Security Specialist Agent (Eve)
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Authorize]
public class FileSyncController : ControllerBase
{
    private readonly ISynchronizationEngine _syncEngine;
    private readonly ILogger<FileSyncController> _logger;

    public FileSyncController(ISynchronizationEngine syncEngine, ILogger<FileSyncController> logger)
    {
        _syncEngine = syncEngine ?? throw new ArgumentNullException(nameof(syncEngine));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Synchronize a specific file with target nodes
    /// </summary>
    /// <param name="request">Synchronization request</param>
    /// <returns>Synchronization result</returns>
    [HttpPost("sync-file")]
    public async Task<ActionResult<SyncResult>> SyncFileAsync([FromBody] SyncFileRequest request)
    {
        try
        {
            _logger.LogInformation("Sync file request received: {FilePath}", request.FilePath);

            if (string.IsNullOrEmpty(request.FilePath))
            {
                return BadRequest("File path is required");
            }

            if (!System.IO.File.Exists(request.FilePath))
            {
                return NotFound($"File not found: {request.FilePath}");
            }

            // Convert target node IDs to SyncNode objects (simplified)
            var targetNodes = request.TargetNodeIds.Select(id => new SyncNode { Id = Guid.Parse(id) });

            var result = await _syncEngine.SynchronizeFileAsync(request.FilePath, targetNodes);

            if (result.Success)
            {
                _logger.LogInformation("File sync completed successfully: {FilePath}", request.FilePath);
                return Ok(result);
            }
            else
            {
                _logger.LogWarning("File sync failed: {FilePath}, Error: {Error}", request.FilePath, result.ErrorMessage);
                return BadRequest(result);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error syncing file: {FilePath}", request.FilePath);
            return StatusCode(500, new { error = "Internal server error", message = ex.Message });
        }
    }

    /// <summary>
    /// Synchronize an entire directory
    /// </summary>
    /// <param name="request">Directory synchronization request</param>
    /// <returns>Synchronization result</returns>
    [HttpPost("sync-directory")]
    public async Task<ActionResult<SyncResult>> SyncDirectoryAsync([FromBody] SyncDirectoryRequest request)
    {
        try
        {
            _logger.LogInformation("Sync directory request received: {DirectoryPath}", request.DirectoryPath);

            if (string.IsNullOrEmpty(request.DirectoryPath))
            {
                return BadRequest("Directory path is required");
            }

            if (!Directory.Exists(request.DirectoryPath))
            {
                return NotFound($"Directory not found: {request.DirectoryPath}");
            }

            var targetNodes = request.TargetNodeIds.Select(id => new SyncNode { Id = Guid.Parse(id) });

            var result = await _syncEngine.SynchronizeDirectoryAsync(
                request.DirectoryPath, 
                targetNodes, 
                request.Recursive);

            if (result.Success)
            {
                _logger.LogInformation("Directory sync completed: {DirectoryPath}, Files: {FileCount}", 
                    request.DirectoryPath, result.FilesSynchronized);
                return Ok(result);
            }
            else
            {
                _logger.LogWarning("Directory sync failed: {DirectoryPath}, Error: {Error}", 
                    request.DirectoryPath, result.ErrorMessage);
                return BadRequest(result);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error syncing directory: {DirectoryPath}", request.DirectoryPath);
            return StatusCode(500, new { error = "Internal server error", message = ex.Message });
        }
    }

    /// <summary>
    /// Get synchronization status for a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>File metadata with sync status</returns>
    [HttpGet("status")]
    public async Task<ActionResult<FileMetadata>> GetFileSyncStatusAsync([FromQuery] string filePath)
    {
        try
        {
            if (string.IsNullOrEmpty(filePath))
            {
                return BadRequest("File path is required");
            }

            var metadata = await _syncEngine.GetFileSyncStatusAsync(filePath);

            if (metadata == null)
            {
                return NotFound($"File not found or not tracked: {filePath}");
            }

            return Ok(metadata);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting file sync status: {FilePath}", filePath);
            return StatusCode(500, new { error = "Internal server error", message = ex.Message });
        }
    }

    /// <summary>
    /// Get all active synchronizations
    /// </summary>
    /// <returns>List of active synchronizations</returns>
    [HttpGet("active")]
    public async Task<ActionResult<IEnumerable<FileMetadata>>> GetActiveSynchronizationsAsync()
    {
        try
        {
            var activeSyncs = await _syncEngine.GetActiveSynchronizationsAsync();
            return Ok(activeSyncs);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting active synchronizations");
            return StatusCode(500, new { error = "Internal server error", message = ex.Message });
        }
    }

    /// <summary>
    /// Resolve a file conflict
    /// </summary>
    /// <param name="request">Conflict resolution request</param>
    /// <returns>Conflict resolution result</returns>
    [HttpPost("resolve-conflict")]
    public async Task<ActionResult<ConflictResolutionResult>> ResolveConflictAsync([FromBody] ResolveConflictRequest request)
    {
        try
        {
            _logger.LogInformation("Conflict resolution request: {FileId}, Strategy: {Strategy}", 
                request.FileId, request.Strategy);

            var result = await _syncEngine.ResolveConflictAsync(request.FileId, request.Strategy);

            if (result.Resolved)
            {
                _logger.LogInformation("Conflict resolved successfully: {FileId}", request.FileId);
                return Ok(result);
            }
            else
            {
                _logger.LogWarning("Conflict resolution failed: {FileId}, Error: {Error}", 
                    request.FileId, result.ErrorMessage);
                return BadRequest(result);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error resolving conflict: {FileId}", request.FileId);
            return StatusCode(500, new { error = "Internal server error", message = ex.Message });
        }
    }

    /// <summary>
    /// Get system health and statistics
    /// </summary>
    /// <returns>System health information</returns>
    [HttpGet("health")]
    [AllowAnonymous]
    public async Task<ActionResult<SystemHealth>> GetSystemHealthAsync()
    {
        try
        {
            var activeSyncs = await _syncEngine.GetActiveSynchronizationsAsync();
            
            var health = new SystemHealth
            {
                Status = "Healthy",
                ActiveSynchronizations = activeSyncs.Count(),
                Uptime = DateTime.UtcNow - Process.GetCurrentProcess().StartTime,
                Version = "1.0.0",
                LastUpdated = DateTime.UtcNow
            };

            return Ok(health);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting system health");
            return StatusCode(500, new { error = "Internal server error", message = ex.Message });
        }
    }
}

// Request/Response DTOs
public class SyncFileRequest
{
    public string FilePath { get; set; } = string.Empty;
    public List<string> TargetNodeIds { get; set; } = new();
    public bool ForceOverwrite { get; set; } = false;
}

public class SyncDirectoryRequest
{
    public string DirectoryPath { get; set; } = string.Empty;
    public List<string> TargetNodeIds { get; set; } = new();
    public bool Recursive { get; set; } = true;
}

public class ResolveConflictRequest
{
    public Guid FileId { get; set; }
    public ConflictResolutionStrategy Strategy { get; set; }
}

public class SystemHealth
{
    public string Status { get; set; } = string.Empty;
    public int ActiveSynchronizations { get; set; }
    public TimeSpan Uptime { get; set; }
    public string Version { get; set; } = string.Empty;
    public DateTime LastUpdated { get; set; }
}
