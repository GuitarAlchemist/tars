using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Records and manages audit trails for changes
/// </summary>
public class AuditTrailService
{
    private readonly ILogger<AuditTrailService> _logger;
    private readonly Dictionary<string, List<AuditEntry>> _auditTrails = new();
    private readonly string _auditDirectory;

    /// <summary>
    /// Initializes a new instance of the <see cref="AuditTrailService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="auditDirectory">The audit directory</param>
    public AuditTrailService(ILogger<AuditTrailService> logger, string auditDirectory = "Audit")
    {
        _logger = logger;
        _auditDirectory = auditDirectory;

        // Create audit directory if it doesn't exist
        if (!Directory.Exists(_auditDirectory))
        {
            Directory.CreateDirectory(_auditDirectory);
        }
    }

    /// <summary>
    /// Creates an audit context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="userId">The user ID</param>
    /// <param name="description">The context description</param>
    /// <returns>True if the audit context was created successfully, false otherwise</returns>
    public bool CreateAuditContext(string contextId, string userId, string description)
    {
        try
        {
            _logger.LogInformation("Creating audit context: {ContextId}", contextId);

            // Check if context already exists
            if (_auditTrails.ContainsKey(contextId))
            {
                _logger.LogWarning("Audit context already exists: {ContextId}", contextId);
                return false;
            }

            // Create audit context
            _auditTrails[contextId] = new List<AuditEntry>();

            // Record context creation
            RecordEntry(contextId, AuditEntryType.ContextCreation, "Context", contextId, userId, description);

            _logger.LogInformation("Created audit context: {ContextId}", contextId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating audit context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Records an audit entry
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="type">The entry type</param>
    /// <param name="targetType">The target type</param>
    /// <param name="targetId">The target ID</param>
    /// <param name="userId">The user ID</param>
    /// <param name="description">The entry description</param>
    /// <param name="details">The entry details</param>
    /// <returns>The audit entry ID</returns>
    public string RecordEntry(
        string contextId,
        AuditEntryType type,
        string targetType,
        string targetId,
        string userId,
        string description,
        string? details = null)
    {
        try
        {
            _logger.LogInformation("Recording audit entry: {Type} on {TargetType} {TargetId}", type, targetType, targetId);

            // Check if context exists
            if (!_auditTrails.TryGetValue(contextId, out var auditTrail))
            {
                _logger.LogWarning("Audit context not found: {ContextId}", contextId);
                return string.Empty;
            }

            // Create audit entry
            var entry = new AuditEntry
            {
                Id = Guid.NewGuid().ToString(),
                ContextId = contextId,
                Type = type,
                TargetType = targetType,
                TargetId = targetId,
                UserId = userId,
                Description = description,
                Details = details,
                Timestamp = DateTime.UtcNow
            };

            // Add entry to audit trail
            auditTrail.Add(entry);

            // Write entry to file
            WriteEntryToFile(entry);

            _logger.LogInformation("Recorded audit entry: {EntryId}", entry.Id);
            return entry.Id;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error recording audit entry: {Type} on {TargetType} {TargetId}", type, targetType, targetId);
            return string.Empty;
        }
    }

    /// <summary>
    /// Gets all audit entries for a context
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>The list of audit entries</returns>
    public List<AuditEntry> GetAuditEntries(string contextId)
    {
        if (_auditTrails.TryGetValue(contextId, out var auditTrail))
        {
            return auditTrail.ToList();
        }
        return new List<AuditEntry>();
    }

    /// <summary>
    /// Gets audit entries for a context filtered by criteria
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="type">The entry type</param>
    /// <param name="targetType">The target type</param>
    /// <param name="targetId">The target ID</param>
    /// <param name="userId">The user ID</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The list of audit entries</returns>
    public List<AuditEntry> GetAuditEntries(
        string contextId,
        AuditEntryType? type = null,
        string? targetType = null,
        string? targetId = null,
        string? userId = null,
        DateTime? startTime = null,
        DateTime? endTime = null)
    {
        if (!_auditTrails.TryGetValue(contextId, out var auditTrail))
        {
            return new List<AuditEntry>();
        }

        var query = auditTrail.AsQueryable();

        if (type != null)
        {
            query = query.Where(e => e.Type == type);
        }

        if (!string.IsNullOrEmpty(targetType))
        {
            query = query.Where(e => e.TargetType == targetType);
        }

        if (!string.IsNullOrEmpty(targetId))
        {
            query = query.Where(e => e.TargetId == targetId);
        }

        if (!string.IsNullOrEmpty(userId))
        {
            query = query.Where(e => e.UserId == userId);
        }

        if (startTime != null)
        {
            query = query.Where(e => e.Timestamp >= startTime);
        }

        if (endTime != null)
        {
            query = query.Where(e => e.Timestamp <= endTime);
        }

        return query.ToList();
    }

    /// <summary>
    /// Exports audit entries to a file
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the audit entries were exported successfully, false otherwise</returns>
    public async Task<bool> ExportAuditEntriesAsync(string contextId, string filePath)
    {
        try
        {
            _logger.LogInformation("Exporting audit entries for context: {ContextId}", contextId);

            // Get audit entries
            var entries = GetAuditEntries(contextId);
            if (entries.Count == 0)
            {
                _logger.LogWarning("No audit entries found for context: {ContextId}", contextId);
                return false;
            }

            // Create directory if it doesn't exist
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Write entries to file
            var json = JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation("Exported {EntryCount} audit entries to {FilePath}", entries.Count, filePath);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error exporting audit entries for context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Imports audit entries from a file
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <param name="filePath">The file path</param>
    /// <returns>True if the audit entries were imported successfully, false otherwise</returns>
    public async Task<bool> ImportAuditEntriesAsync(string contextId, string filePath)
    {
        try
        {
            _logger.LogInformation("Importing audit entries for context: {ContextId}", contextId);

            // Check if file exists
            if (!File.Exists(filePath))
            {
                _logger.LogWarning("Audit file not found: {FilePath}", filePath);
                return false;
            }

            // Read entries from file
            var json = await File.ReadAllTextAsync(filePath);
            var entries = JsonSerializer.Deserialize<List<AuditEntry>>(json);
            if (entries == null || entries.Count == 0)
            {
                _logger.LogWarning("No audit entries found in file: {FilePath}", filePath);
                return false;
            }

            // Create audit context if it doesn't exist
            if (!_auditTrails.ContainsKey(contextId))
            {
                _auditTrails[contextId] = new List<AuditEntry>();
            }

            // Add entries to audit trail
            foreach (var entry in entries)
            {
                entry.ContextId = contextId;
                _auditTrails[contextId].Add(entry);
            }

            _logger.LogInformation("Imported {EntryCount} audit entries from {FilePath}", entries.Count, filePath);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error importing audit entries for context: {ContextId}", contextId);
            return false;
        }
    }

    /// <summary>
    /// Writes an audit entry to a file
    /// </summary>
    /// <param name="entry">The audit entry</param>
    private void WriteEntryToFile(AuditEntry entry)
    {
        try
        {
            // Create context directory if it doesn't exist
            var contextDirectory = Path.Combine(_auditDirectory, entry.ContextId);
            if (!Directory.Exists(contextDirectory))
            {
                Directory.CreateDirectory(contextDirectory);
            }

            // Create file path
            var fileName = $"{entry.Timestamp:yyyyMMddHHmmss}_{entry.Type}_{entry.Id}.json";
            var filePath = Path.Combine(contextDirectory, fileName);

            // Write entry to file
            var json = JsonSerializer.Serialize(entry, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error writing audit entry to file: {EntryId}", entry.Id);
        }
    }

    /// <summary>
    /// Gets the available audit options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "AuditDirectory", "The directory where audit entries are stored" },
            { "EnableFileAudit", "Whether to write audit entries to files (true, false)" },
            { "EnableDetailedAudit", "Whether to record detailed audit information (true, false)" }
        };
    }
}
