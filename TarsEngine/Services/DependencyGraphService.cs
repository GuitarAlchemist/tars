using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for managing improvement dependency graphs
/// </summary>
public class DependencyGraphService
{
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="DependencyGraphService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public DependencyGraphService(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Creates a dependency graph from a list of improvements
    /// </summary>
    /// <param name="improvements">The improvements</param>
    /// <param name="options">Optional graph options</param>
    /// <returns>The dependency graph</returns>
    public ImprovementDependencyGraph CreateDependencyGraph(
        List<PrioritizedImprovement> improvements,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Creating dependency graph for {ImprovementCount} improvements", improvements.Count);

            var graph = new ImprovementDependencyGraph();

            // Create nodes
            foreach (var improvement in improvements)
            {
                var node = new ImprovementNode
                {
                    Id = improvement.Id,
                    Name = improvement.Name,
                    Category = improvement.Category,
                    PriorityScore = improvement.PriorityScore,
                    Status = improvement.Status,
                    Metadata = new Dictionary<string, string>
                    {
                        { "Description", improvement.Description },
                        { "ImpactScore", improvement.ImpactScore.ToString() },
                        { "EffortScore", improvement.EffortScore.ToString() },
                        { "RiskScore", improvement.RiskScore.ToString() },
                        { "AlignmentScore", improvement.AlignmentScore.ToString() }
                    }
                };

                graph.Nodes.Add(node);
            }

            // Create edges
            foreach (var improvement in improvements)
            {
                foreach (var dependencyId in improvement.Dependencies)
                {
                    if (graph.Nodes.Any(n => n.Id == dependencyId))
                    {
                        var edge = new ImprovementEdge
                        {
                            Id = $"{improvement.Id}-{dependencyId}",
                            SourceId = improvement.Id,
                            TargetId = dependencyId,
                            Type = ImprovementDependencyType.Requires,
                            Weight = 1.0
                        };

                        graph.Edges.Add(edge);
                    }
                }
            }

            // Detect file-based dependencies
            if (ParseOption(options, "DetectFileDependencies", true))
            {
                DetectFileDependencies(improvements, graph);
            }

            // Detect category-based dependencies
            if (ParseOption(options, "DetectCategoryDependencies", true))
            {
                DetectCategoryDependencies(improvements, graph);
            }

            // Detect tag-based dependencies
            if (ParseOption(options, "DetectTagDependencies", true))
            {
                DetectTagDependencies(improvements, graph);
            }

            // Detect cycles
            var cycles = graph.DetectCycles();
            if (cycles.Count > 0)
            {
                _logger.LogWarning("Detected {CycleCount} cycles in dependency graph", cycles.Count);

                // Break cycles if requested
                if (ParseOption(options, "BreakCycles", true))
                {
                    BreakCycles(graph, cycles);
                }
            }

            _logger.LogInformation("Created dependency graph with {NodeCount} nodes and {EdgeCount} edges", graph.Nodes.Count, graph.Edges.Count);
            return graph;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating dependency graph");
            return new ImprovementDependencyGraph();
        }
    }

    /// <summary>
    /// Updates a dependency graph with new improvements
    /// </summary>
    /// <param name="graph">The dependency graph</param>
    /// <param name="improvements">The improvements</param>
    /// <param name="options">Optional update options</param>
    /// <returns>The updated dependency graph</returns>
    public ImprovementDependencyGraph UpdateDependencyGraph(
        ImprovementDependencyGraph graph,
        List<PrioritizedImprovement> improvements,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Updating dependency graph with {ImprovementCount} improvements", improvements.Count);

            // Add new nodes
            foreach (var improvement in improvements)
            {
                if (!graph.Nodes.Any(n => n.Id == improvement.Id))
                {
                    var node = new ImprovementNode
                    {
                        Id = improvement.Id,
                        Name = improvement.Name,
                        Category = improvement.Category,
                        PriorityScore = improvement.PriorityScore,
                        Status = improvement.Status,
                        Metadata = new Dictionary<string, string>
                        {
                            { "Description", improvement.Description },
                            { "ImpactScore", improvement.ImpactScore.ToString() },
                            { "EffortScore", improvement.EffortScore.ToString() },
                            { "RiskScore", improvement.RiskScore.ToString() },
                            { "AlignmentScore", improvement.AlignmentScore.ToString() }
                        }
                    };

                    graph.Nodes.Add(node);
                }
                else
                {
                    // Update existing node
                    var node = graph.Nodes.First(n => n.Id == improvement.Id);
                    node.Name = improvement.Name;
                    node.Category = improvement.Category;
                    node.PriorityScore = improvement.PriorityScore;
                    node.Status = improvement.Status;
                    node.Metadata["Description"] = improvement.Description;
                    node.Metadata["ImpactScore"] = improvement.ImpactScore.ToString();
                    node.Metadata["EffortScore"] = improvement.EffortScore.ToString();
                    node.Metadata["RiskScore"] = improvement.RiskScore.ToString();
                    node.Metadata["AlignmentScore"] = improvement.AlignmentScore.ToString();
                }
            }

            // Add new edges
            foreach (var improvement in improvements)
            {
                // Remove existing edges
                graph.Edges.RemoveAll(e => e.SourceId == improvement.Id);

                // Add new edges
                foreach (var dependencyId in improvement.Dependencies)
                {
                    if (graph.Nodes.Any(n => n.Id == dependencyId))
                    {
                        var edge = new ImprovementEdge
                        {
                            Id = $"{improvement.Id}-{dependencyId}",
                            SourceId = improvement.Id,
                            TargetId = dependencyId,
                            Type = ImprovementDependencyType.Requires,
                            Weight = 1.0
                        };

                        graph.Edges.Add(edge);
                    }
                }
            }

            // Detect file-based dependencies
            if (ParseOption(options, "DetectFileDependencies", true))
            {
                DetectFileDependencies(improvements, graph);
            }

            // Detect category-based dependencies
            if (ParseOption(options, "DetectCategoryDependencies", true))
            {
                DetectCategoryDependencies(improvements, graph);
            }

            // Detect tag-based dependencies
            if (ParseOption(options, "DetectTagDependencies", true))
            {
                DetectTagDependencies(improvements, graph);
            }

            // Detect cycles
            var cycles = graph.DetectCycles();
            if (cycles.Count > 0)
            {
                _logger.LogWarning("Detected {CycleCount} cycles in dependency graph", cycles.Count);

                // Break cycles if requested
                if (ParseOption(options, "BreakCycles", true))
                {
                    BreakCycles(graph, cycles);
                }
            }

            _logger.LogInformation("Updated dependency graph with {NodeCount} nodes and {EdgeCount} edges", graph.Nodes.Count, graph.Edges.Count);
            return graph;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating dependency graph");
            return graph;
        }
    }

    /// <summary>
    /// Gets the next improvements to implement
    /// </summary>
    /// <param name="graph">The dependency graph</param>
    /// <param name="count">The number of improvements to get</param>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of improvements</returns>
    public List<ImprovementNode> GetNextImprovements(
        ImprovementDependencyGraph graph,
        int count,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting next {Count} improvements from dependency graph", count);

            // Get available improvements (no dependencies or all dependencies completed)
            var availableImprovements = new List<ImprovementNode>();
            foreach (var node in graph.Nodes)
            {
                if (node.Status == ImprovementStatus.Pending || node.Status == ImprovementStatus.Approved)
                {
                    var dependencies = graph.GetDependencies(node.Id);
                    var allDependenciesCompleted = dependencies.All(d => 
                        d.Status == ImprovementStatus.Completed || 
                        d.Status == ImprovementStatus.Merged || 
                        d.Status == ImprovementStatus.Deployed);

                    if (allDependenciesCompleted)
                    {
                        availableImprovements.Add(node);
                    }
                }
            }

            // Sort by priority score (descending)
            availableImprovements = availableImprovements
                .OrderByDescending(i => i.PriorityScore)
                .ToList();

            // Apply category filter
            if (options != null && options.TryGetValue("Category", out var category))
            {
                if (Enum.TryParse<ImprovementCategory>(category, true, out var categoryEnum))
                {
                    availableImprovements = availableImprovements
                        .Where(i => i.Category == categoryEnum)
                        .ToList();
                }
            }

            // Apply status filter
            if (options != null && options.TryGetValue("Status", out var status))
            {
                if (Enum.TryParse<ImprovementStatus>(status, true, out var statusEnum))
                {
                    availableImprovements = availableImprovements
                        .Where(i => i.Status == statusEnum)
                        .ToList();
                }
            }

            // Apply minimum priority filter
            if (options != null && options.TryGetValue("MinPriority", out var minPriorityStr))
            {
                if (double.TryParse(minPriorityStr, out var minPriority))
                {
                    availableImprovements = availableImprovements
                        .Where(i => i.PriorityScore >= minPriority)
                        .ToList();
                }
            }

            // Take the requested number of improvements
            var result = availableImprovements.Take(count).ToList();

            _logger.LogInformation("Got {ResultCount} next improvements from dependency graph", result.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting next improvements from dependency graph");
            return [];
        }
    }

    /// <summary>
    /// Gets the available graph options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "DetectFileDependencies", "Whether to detect file-based dependencies (default: true)" },
            { "DetectCategoryDependencies", "Whether to detect category-based dependencies (default: true)" },
            { "DetectTagDependencies", "Whether to detect tag-based dependencies (default: true)" },
            { "BreakCycles", "Whether to break cycles in the dependency graph (default: true)" },
            { "Category", "Filter improvements by category" },
            { "Status", "Filter improvements by status" },
            { "MinPriority", "Minimum priority score for improvements" }
        };
    }

    private void DetectFileDependencies(List<PrioritizedImprovement> improvements, ImprovementDependencyGraph graph)
    {
        try
        {
            _logger.LogInformation("Detecting file-based dependencies");

            // Group improvements by affected files
            var improvementsByFile = new Dictionary<string, List<string>>();
            foreach (var improvement in improvements)
            {
                foreach (var file in improvement.AffectedFiles)
                {
                    if (!improvementsByFile.ContainsKey(file))
                    {
                        improvementsByFile[file] = [];
                    }
                    improvementsByFile[file].Add(improvement.Id);
                }
            }

            // Create edges for improvements affecting the same files
            foreach (var file in improvementsByFile.Keys)
            {
                var improvementIds = improvementsByFile[file];
                if (improvementIds.Count > 1)
                {
                    for (int i = 0; i < improvementIds.Count; i++)
                    {
                        for (int j = i + 1; j < improvementIds.Count; j++)
                        {
                            var sourceId = improvementIds[i];
                            var targetId = improvementIds[j];

                            // Skip if there's already a direct dependency
                            if (graph.Edges.Any(e => e.SourceId == sourceId && e.TargetId == targetId) ||
                                graph.Edges.Any(e => e.SourceId == targetId && e.TargetId == sourceId))
                            {
                                continue;
                            }

                            // Create a "related to" edge
                            var edge = new ImprovementEdge
                            {
                                Id = $"{sourceId}-{targetId}-file",
                                SourceId = sourceId,
                                TargetId = targetId,
                                Type = ImprovementDependencyType.RelatedTo,
                                Weight = 0.5,
                                Metadata = new Dictionary<string, string>
                                {
                                    { "File", file }
                                }
                            };

                            graph.Edges.Add(edge);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting file-based dependencies");
        }
    }

    private void DetectCategoryDependencies(List<PrioritizedImprovement> improvements, ImprovementDependencyGraph graph)
    {
        try
        {
            _logger.LogInformation("Detecting category-based dependencies");

            // Define category dependencies
            var categoryDependencies = new Dictionary<ImprovementCategory, List<ImprovementCategory>>
            {
                { ImprovementCategory.Security, [ImprovementCategory.Architecture] },
                { ImprovementCategory.Performance, [ImprovementCategory.Architecture] },
                { ImprovementCategory.Scalability, [ImprovementCategory.Architecture, ImprovementCategory.Performance] },
                { ImprovementCategory.Reliability, [ImprovementCategory.Architecture, ImprovementCategory.Security] },
                { ImprovementCategory.Maintainability,
                    [ImprovementCategory.CodeQuality, ImprovementCategory.Testability]
                },
                { ImprovementCategory.Testability, [ImprovementCategory.Architecture] },
                { ImprovementCategory.Usability, [ImprovementCategory.Functionality] },
                { ImprovementCategory.Accessibility, [ImprovementCategory.Usability] },
                { ImprovementCategory.Internationalization, [ImprovementCategory.Architecture] },
                { ImprovementCategory.Localization, [ImprovementCategory.Internationalization] },
                { ImprovementCategory.Compatibility, [ImprovementCategory.Architecture] },
                { ImprovementCategory.Portability, [ImprovementCategory.Architecture] },
                { ImprovementCategory.Extensibility, [ImprovementCategory.Architecture] },
                { ImprovementCategory.Reusability, [ImprovementCategory.Architecture, ImprovementCategory.Modularity] },
                { ImprovementCategory.Modularity, [ImprovementCategory.Architecture] }
            };

            // Group improvements by category
            var improvementsByCategory = new Dictionary<ImprovementCategory, List<string>>();
            foreach (var improvement in improvements)
            {
                if (!improvementsByCategory.ContainsKey(improvement.Category))
                {
                    improvementsByCategory[improvement.Category] = [];
                }
                improvementsByCategory[improvement.Category].Add(improvement.Id);
            }

            // Create edges for category dependencies
            foreach (var category in categoryDependencies.Keys)
            {
                if (!improvementsByCategory.ContainsKey(category))
                {
                    continue;
                }

                foreach (var dependencyCategory in categoryDependencies[category])
                {
                    if (!improvementsByCategory.ContainsKey(dependencyCategory))
                    {
                        continue;
                    }

                    foreach (var sourceId in improvementsByCategory[category])
                    {
                        foreach (var targetId in improvementsByCategory[dependencyCategory])
                        {
                            // Skip if there's already a direct dependency
                            if (graph.Edges.Any(e => e.SourceId == sourceId && e.TargetId == targetId))
                            {
                                continue;
                            }

                            // Create a "related to" edge
                            var edge = new ImprovementEdge
                            {
                                Id = $"{sourceId}-{targetId}-category",
                                SourceId = sourceId,
                                TargetId = targetId,
                                Type = ImprovementDependencyType.RelatedTo,
                                Weight = 0.3,
                                Metadata = new Dictionary<string, string>
                                {
                                    { "Category", category.ToString() },
                                    { "DependencyCategory", dependencyCategory.ToString() }
                                }
                            };

                            graph.Edges.Add(edge);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting category-based dependencies");
        }
    }

    private void DetectTagDependencies(List<PrioritizedImprovement> improvements, ImprovementDependencyGraph graph)
    {
        try
        {
            _logger.LogInformation("Detecting tag-based dependencies");

            // Group improvements by tag
            var improvementsByTag = new Dictionary<string, List<string>>();
            foreach (var improvement in improvements)
            {
                foreach (var tag in improvement.Tags)
                {
                    if (!improvementsByTag.ContainsKey(tag))
                    {
                        improvementsByTag[tag] = [];
                    }
                    improvementsByTag[tag].Add(improvement.Id);
                }
            }

            // Create edges for improvements with the same tags
            foreach (var tag in improvementsByTag.Keys)
            {
                var improvementIds = improvementsByTag[tag];
                if (improvementIds.Count > 1)
                {
                    for (int i = 0; i < improvementIds.Count; i++)
                    {
                        for (int j = i + 1; j < improvementIds.Count; j++)
                        {
                            var sourceId = improvementIds[i];
                            var targetId = improvementIds[j];

                            // Skip if there's already a direct dependency
                            if (graph.Edges.Any(e => e.SourceId == sourceId && e.TargetId == targetId) ||
                                graph.Edges.Any(e => e.SourceId == targetId && e.TargetId == sourceId))
                            {
                                continue;
                            }

                            // Create a "related to" edge
                            var edge = new ImprovementEdge
                            {
                                Id = $"{sourceId}-{targetId}-tag",
                                SourceId = sourceId,
                                TargetId = targetId,
                                Type = ImprovementDependencyType.RelatedTo,
                                Weight = 0.2,
                                Metadata = new Dictionary<string, string>
                                {
                                    { "Tag", tag }
                                }
                            };

                            graph.Edges.Add(edge);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting tag-based dependencies");
        }
    }

    private void BreakCycles(ImprovementDependencyGraph graph, List<List<ImprovementNode>> cycles)
    {
        try
        {
            _logger.LogInformation("Breaking {CycleCount} cycles in dependency graph", cycles.Count);

            foreach (var cycle in cycles)
            {
                if (cycle.Count < 2)
                {
                    continue;
                }

                // Find the edge with the lowest weight
                var lowestWeight = double.MaxValue;
                ImprovementEdge? edgeToRemove = null;

                for (int i = 0; i < cycle.Count; i++)
                {
                    var sourceId = cycle[i].Id;
                    var targetId = cycle[(i + 1) % cycle.Count].Id;

                    var edge = graph.Edges.FirstOrDefault(e => e.SourceId == sourceId && e.TargetId == targetId);
                    if (edge != null && edge.Weight < lowestWeight)
                    {
                        lowestWeight = edge.Weight;
                        edgeToRemove = edge;
                    }
                }

                // Remove the edge with the lowest weight
                if (edgeToRemove != null)
                {
                    _logger.LogInformation("Breaking cycle by removing edge: {EdgeId}", edgeToRemove.Id);
                    graph.Edges.Remove(edgeToRemove);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error breaking cycles in dependency graph");
        }
    }

    private T ParseOption<T>(Dictionary<string, string>? options, string key, T defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        try
        {
            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return defaultValue;
        }
    }
}
