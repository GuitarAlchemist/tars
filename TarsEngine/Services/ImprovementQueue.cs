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
/// Service for managing the improvement queue
/// </summary>
public class ImprovementQueue
{
    private readonly ILogger _logger;
    private readonly string _queueDirectory;
    private readonly Dictionary<string, PrioritizedImprovement> _improvements = new();
    private readonly DependencyGraphService _dependencyGraphService;
    private ImprovementDependencyGraph? _dependencyGraph;
    private bool _isInitialized = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImprovementQueue"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="queueDirectory">The directory for storing improvements</param>
    /// <param name="dependencyGraphService">The dependency graph service</param>
    public ImprovementQueue(ILogger logger, string queueDirectory, DependencyGraphService dependencyGraphService)
    {
        _logger = logger;
        _queueDirectory = queueDirectory;
        _dependencyGraphService = dependencyGraphService;
    }

    /// <summary>
    /// Initializes the improvement queue
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing improvement queue from directory: {QueueDirectory}", _queueDirectory);

            if (!Directory.Exists(_queueDirectory))
            {
                _logger.LogWarning("Queue directory not found: {QueueDirectory}", _queueDirectory);
                Directory.CreateDirectory(_queueDirectory);
            }

            // Clear existing improvements
            _improvements.Clear();

            // Load improvements from files
            var improvementFiles = Directory.GetFiles(_queueDirectory, "*.json", SearchOption.AllDirectories);
            _logger.LogInformation("Found {ImprovementFileCount} improvement files", improvementFiles.Length);

            foreach (var file in improvementFiles)
            {
                try
                {
                    var json = await File.ReadAllTextAsync(file);
                    var improvement = JsonSerializer.Deserialize<PrioritizedImprovement>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                    if (improvement != null)
                    {
                        _improvements[improvement.Id] = improvement;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading improvement file: {FilePath}", file);
                }
            }

            // Create dependency graph
            _dependencyGraph = _dependencyGraphService.CreateDependencyGraph(_improvements.Values.ToList());

            _logger.LogInformation("Initialized improvement queue with {ImprovementCount} improvements", _improvements.Count);
            _isInitialized = true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing improvement queue");
        }
    }

    /// <summary>
    /// Gets all improvements
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of improvements</returns>
    public async Task<List<PrioritizedImprovement>> GetImprovementsAsync(Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting improvements");

            // Apply filters
            var improvements = _improvements.Values.ToList();

            if (options != null)
            {
                // Filter by category
                if (options.TryGetValue("Category", out var category) && Enum.TryParse<ImprovementCategory>(category, true, out var categoryEnum))
                {
                    improvements = improvements.Where(i => i.Category == categoryEnum).ToList();
                }

                // Filter by status
                if (options.TryGetValue("Status", out var status) && Enum.TryParse<ImprovementStatus>(status, true, out var statusEnum))
                {
                    improvements = improvements.Where(i => i.Status == statusEnum).ToList();
                }

                // Filter by tag
                if (options.TryGetValue("Tag", out var tag) && !string.IsNullOrEmpty(tag))
                {
                    improvements = improvements.Where(i => i.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase)).ToList();
                }

                // Filter by affected file
                if (options.TryGetValue("AffectedFile", out var affectedFile) && !string.IsNullOrEmpty(affectedFile))
                {
                    improvements = improvements.Where(i => i.AffectedFiles.Contains(affectedFile)).ToList();
                }

                // Filter by minimum priority
                if (options.TryGetValue("MinPriority", out var minPriorityStr) && double.TryParse(minPriorityStr, out var minPriority))
                {
                    improvements = improvements.Where(i => i.PriorityScore >= minPriority).ToList();
                }

                // Filter by maximum priority
                if (options.TryGetValue("MaxPriority", out var maxPriorityStr) && double.TryParse(maxPriorityStr, out var maxPriority))
                {
                    improvements = improvements.Where(i => i.PriorityScore <= maxPriority).ToList();
                }

                // Filter by strategic goal
                if (options.TryGetValue("StrategicGoal", out var strategicGoal) && !string.IsNullOrEmpty(strategicGoal))
                {
                    improvements = improvements.Where(i => i.StrategicGoals.Contains(strategicGoal)).ToList();
                }

                // Sort by priority
                if (options.TryGetValue("SortBy", out var sortBy) && !string.IsNullOrEmpty(sortBy))
                {
                    switch (sortBy.ToLowerInvariant())
                    {
                        case "priority":
                            improvements = improvements.OrderByDescending(i => i.PriorityScore).ToList();
                            break;
                        case "impact":
                            improvements = improvements.OrderByDescending(i => i.ImpactScore).ToList();
                            break;
                        case "effort":
                            improvements = improvements.OrderBy(i => i.EffortScore).ToList();
                            break;
                        case "risk":
                            improvements = improvements.OrderBy(i => i.RiskScore).ToList();
                            break;
                        case "alignment":
                            improvements = improvements.OrderByDescending(i => i.AlignmentScore).ToList();
                            break;
                        case "date":
                            improvements = improvements.OrderByDescending(i => i.CreatedAt).ToList();
                            break;
                        case "name":
                            improvements = improvements.OrderBy(i => i.Name).ToList();
                            break;
                    }
                }

                // Limit results
                if (options.TryGetValue("Limit", out var limitStr) && int.TryParse(limitStr, out var limit) && limit > 0)
                {
                    improvements = improvements.Take(limit).ToList();
                }
            }

            _logger.LogInformation("Got {ImprovementCount} improvements", improvements.Count);
            return improvements;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting improvements");
            return new List<PrioritizedImprovement>();
        }
    }

    /// <summary>
    /// Gets an improvement by ID
    /// </summary>
    /// <param name="improvementId">The improvement ID</param>
    /// <returns>The improvement, or null if not found</returns>
    public async Task<PrioritizedImprovement?> GetImprovementAsync(string improvementId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting improvement by ID: {ImprovementId}", improvementId);

            if (_improvements.TryGetValue(improvementId, out var improvement))
            {
                return improvement;
            }

            _logger.LogWarning("Improvement not found: {ImprovementId}", improvementId);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting improvement: {ImprovementId}", improvementId);
            return null;
        }
    }

    /// <summary>
    /// Adds an improvement
    /// </summary>
    /// <param name="improvement">The improvement to add</param>
    /// <returns>True if the improvement was added successfully, false otherwise</returns>
    public async Task<bool> AddImprovementAsync(PrioritizedImprovement improvement)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Adding improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);

            if (_improvements.ContainsKey(improvement.Id))
            {
                _logger.LogWarning("Improvement already exists: {ImprovementId}", improvement.Id);
                return false;
            }

            // Add improvement to memory
            _improvements[improvement.Id] = improvement;

            // Save improvement to file
            await SaveImprovementToFileAsync(improvement);

            // Update dependency graph
            if (_dependencyGraph != null)
            {
                _dependencyGraph = _dependencyGraphService.UpdateDependencyGraph(_dependencyGraph, new List<PrioritizedImprovement> { improvement });
            }
            else
            {
                _dependencyGraph = _dependencyGraphService.CreateDependencyGraph(_improvements.Values.ToList());
            }

            _logger.LogInformation("Added improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return false;
        }
    }

    /// <summary>
    /// Updates an improvement
    /// </summary>
    /// <param name="improvement">The improvement to update</param>
    /// <returns>True if the improvement was updated successfully, false otherwise</returns>
    public async Task<bool> UpdateImprovementAsync(PrioritizedImprovement improvement)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Updating improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);

            if (!_improvements.ContainsKey(improvement.Id))
            {
                _logger.LogWarning("Improvement not found: {ImprovementId}", improvement.Id);
                return false;
            }

            // Update improvement in memory
            improvement.UpdatedAt = DateTime.UtcNow;
            _improvements[improvement.Id] = improvement;

            // Save improvement to file
            await SaveImprovementToFileAsync(improvement);

            // Update dependency graph
            if (_dependencyGraph != null)
            {
                _dependencyGraph = _dependencyGraphService.UpdateDependencyGraph(_dependencyGraph, new List<PrioritizedImprovement> { improvement });
            }
            else
            {
                _dependencyGraph = _dependencyGraphService.CreateDependencyGraph(_improvements.Values.ToList());
            }

            _logger.LogInformation("Updated improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return false;
        }
    }

    /// <summary>
    /// Removes an improvement
    /// </summary>
    /// <param name="improvementId">The ID of the improvement to remove</param>
    /// <returns>True if the improvement was removed successfully, false otherwise</returns>
    public async Task<bool> RemoveImprovementAsync(string improvementId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Removing improvement: {ImprovementId}", improvementId);

            if (!_improvements.TryGetValue(improvementId, out var improvement))
            {
                _logger.LogWarning("Improvement not found: {ImprovementId}", improvementId);
                return false;
            }

            // Remove improvement from memory
            _improvements.Remove(improvementId);

            // Remove improvement file
            var filePath = GetImprovementFilePath(improvement);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }

            // Update dependency graph
            _dependencyGraph = _dependencyGraphService.CreateDependencyGraph(_improvements.Values.ToList());

            _logger.LogInformation("Removed improvement: {ImprovementId}", improvementId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing improvement: {ImprovementId}", improvementId);
            return false;
        }
    }

    /// <summary>
    /// Gets the next improvements to implement
    /// </summary>
    /// <param name="count">The number of improvements to get</param>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of improvements</returns>
    public async Task<List<PrioritizedImprovement>> GetNextImprovementsAsync(int count, Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting next {Count} improvements", count);

            if (_dependencyGraph == null)
            {
                _dependencyGraph = _dependencyGraphService.CreateDependencyGraph(_improvements.Values.ToList());
            }

            // Get next improvements from dependency graph
            var nextNodes = _dependencyGraphService.GetNextImprovements(_dependencyGraph, count, options);

            // Convert nodes to improvements
            var nextImprovements = new List<PrioritizedImprovement>();
            foreach (var node in nextNodes)
            {
                if (_improvements.TryGetValue(node.Id, out var improvement))
                {
                    nextImprovements.Add(improvement);
                }
            }

            _logger.LogInformation("Got {ImprovementCount} next improvements", nextImprovements.Count);
            return nextImprovements;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting next improvements");
            return new List<PrioritizedImprovement>();
        }
    }

    /// <summary>
    /// Gets the dependency graph
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The dependency graph</returns>
    public async Task<ImprovementDependencyGraph> GetDependencyGraphAsync(Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting dependency graph");

            if (_dependencyGraph == null)
            {
                _dependencyGraph = _dependencyGraphService.CreateDependencyGraph(_improvements.Values.ToList(), options);
            }

            return _dependencyGraph;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting dependency graph");
            return new ImprovementDependencyGraph();
        }
    }

    /// <summary>
    /// Gets the available queue options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "Category", "Filter improvements by category" },
            { "Status", "Filter improvements by status" },
            { "Tag", "Filter improvements by tag" },
            { "AffectedFile", "Filter improvements by affected file" },
            { "MinPriority", "Minimum priority score for improvements" },
            { "MaxPriority", "Maximum priority score for improvements" },
            { "StrategicGoal", "Filter improvements by strategic goal" },
            { "SortBy", "Sort improvements by priority, impact, effort, risk, alignment, date, or name" },
            { "Limit", "Limit the number of improvements returned" },
            { "DetectFileDependencies", "Whether to detect file-based dependencies (default: true)" },
            { "DetectCategoryDependencies", "Whether to detect category-based dependencies (default: true)" },
            { "DetectTagDependencies", "Whether to detect tag-based dependencies (default: true)" },
            { "BreakCycles", "Whether to break cycles in the dependency graph (default: true)" }
        };
    }

    private async Task SaveImprovementToFileAsync(PrioritizedImprovement improvement)
    {
        var filePath = GetImprovementFilePath(improvement);
        var directory = Path.GetDirectoryName(filePath);
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory!);
        }

        var json = JsonSerializer.Serialize(improvement, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(filePath, json);
    }

    private string GetImprovementFilePath(PrioritizedImprovement improvement)
    {
        var category = improvement.Category.ToString();
        var fileName = $"{improvement.Id}.json";

        return Path.Combine(_queueDirectory, category, fileName);
    }

    private async Task EnsureInitializedAsync()
    {
        if (!_isInitialized)
        {
            await InitializeAsync();
        }
    }
}
