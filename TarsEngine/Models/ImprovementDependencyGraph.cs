using System;
using System.Collections.Generic;
using System.Linq;

namespace TarsEngine.Models;

/// <summary>
/// Represents a dependency graph for improvements
/// </summary>
public class ImprovementDependencyGraph
{
    /// <summary>
    /// Gets or sets the nodes in the graph
    /// </summary>
    public List<ImprovementNode> Nodes { get; set; } = new List<ImprovementNode>();

    /// <summary>
    /// Gets or sets the edges in the graph
    /// </summary>
    public List<ImprovementEdge> Edges { get; set; } = new List<ImprovementEdge>();

    /// <summary>
    /// Gets or sets the timestamp when the graph was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets the root nodes in the graph
    /// </summary>
    public List<ImprovementNode> RootNodes => Nodes.Where(n => !Edges.Any(e => e.TargetId == n.Id)).ToList();

    /// <summary>
    /// Gets the leaf nodes in the graph
    /// </summary>
    public List<ImprovementNode> LeafNodes => Nodes.Where(n => !Edges.Any(e => e.SourceId == n.Id)).ToList();

    /// <summary>
    /// Gets the dependencies for a node
    /// </summary>
    /// <param name="nodeId">The node ID</param>
    /// <returns>The list of dependencies</returns>
    public List<ImprovementNode> GetDependencies(string nodeId)
    {
        var dependencyIds = Edges
            .Where(e => e.SourceId == nodeId)
            .Select(e => e.TargetId)
            .ToList();

        return Nodes
            .Where(n => dependencyIds.Contains(n.Id))
            .ToList();
    }

    /// <summary>
    /// Gets the dependents for a node
    /// </summary>
    /// <param name="nodeId">The node ID</param>
    /// <returns>The list of dependents</returns>
    public List<ImprovementNode> GetDependents(string nodeId)
    {
        var dependentIds = Edges
            .Where(e => e.TargetId == nodeId)
            .Select(e => e.SourceId)
            .ToList();

        return Nodes
            .Where(n => dependentIds.Contains(n.Id))
            .ToList();
    }

    /// <summary>
    /// Gets the transitive dependencies for a node
    /// </summary>
    /// <param name="nodeId">The node ID</param>
    /// <returns>The list of transitive dependencies</returns>
    public List<ImprovementNode> GetTransitiveDependencies(string nodeId)
    {
        var visited = new HashSet<string>();
        var result = new List<ImprovementNode>();
        GetTransitiveDependenciesRecursive(nodeId, visited, result);
        return result;
    }

    /// <summary>
    /// Gets the transitive dependents for a node
    /// </summary>
    /// <param name="nodeId">The node ID</param>
    /// <returns>The list of transitive dependents</returns>
    public List<ImprovementNode> GetTransitiveDependents(string nodeId)
    {
        var visited = new HashSet<string>();
        var result = new List<ImprovementNode>();
        GetTransitiveDependentsRecursive(nodeId, visited, result);
        return result;
    }

    /// <summary>
    /// Gets a topological sort of the graph
    /// </summary>
    /// <returns>The topologically sorted list of nodes</returns>
    public List<ImprovementNode> GetTopologicalSort()
    {
        var result = new List<ImprovementNode>();
        var visited = new HashSet<string>();
        var temp = new HashSet<string>();

        foreach (var node in Nodes)
        {
            if (!visited.Contains(node.Id))
            {
                TopologicalSortRecursive(node.Id, visited, temp, result);
            }
        }

        result.Reverse();
        return result;
    }

    /// <summary>
    /// Detects cycles in the graph
    /// </summary>
    /// <returns>The list of cycles</returns>
    public List<List<ImprovementNode>> DetectCycles()
    {
        var cycles = new List<List<ImprovementNode>>();
        var visited = new HashSet<string>();
        var path = new List<string>();

        foreach (var node in Nodes)
        {
            if (!visited.Contains(node.Id))
            {
                DetectCyclesRecursive(node.Id, visited, path, cycles);
            }
        }

        return cycles;
    }

    private void GetTransitiveDependenciesRecursive(string nodeId, HashSet<string> visited, List<ImprovementNode> result)
    {
        if (visited.Contains(nodeId))
        {
            return;
        }

        visited.Add(nodeId);

        var dependencies = GetDependencies(nodeId);
        foreach (var dependency in dependencies)
        {
            result.Add(dependency);
            GetTransitiveDependenciesRecursive(dependency.Id, visited, result);
        }
    }

    private void GetTransitiveDependentsRecursive(string nodeId, HashSet<string> visited, List<ImprovementNode> result)
    {
        if (visited.Contains(nodeId))
        {
            return;
        }

        visited.Add(nodeId);

        var dependents = GetDependents(nodeId);
        foreach (var dependent in dependents)
        {
            result.Add(dependent);
            GetTransitiveDependentsRecursive(dependent.Id, visited, result);
        }
    }

    private void TopologicalSortRecursive(string nodeId, HashSet<string> visited, HashSet<string> temp, List<ImprovementNode> result)
    {
        if (temp.Contains(nodeId))
        {
            // Cycle detected
            return;
        }

        if (visited.Contains(nodeId))
        {
            return;
        }

        temp.Add(nodeId);

        var dependencies = GetDependencies(nodeId);
        foreach (var dependency in dependencies)
        {
            TopologicalSortRecursive(dependency.Id, visited, temp, result);
        }

        temp.Remove(nodeId);
        visited.Add(nodeId);

        var node = Nodes.FirstOrDefault(n => n.Id == nodeId);
        if (node != null)
        {
            result.Add(node);
        }
    }

    private void DetectCyclesRecursive(string nodeId, HashSet<string> visited, List<string> path, List<List<ImprovementNode>> cycles)
    {
        if (path.Contains(nodeId))
        {
            // Cycle detected
            var cycleStart = path.IndexOf(nodeId);
            var cycle = path.Skip(cycleStart).ToList();
            cycle.Add(nodeId);

            var cycleNodes = cycle
                .Select(id => Nodes.FirstOrDefault(n => n.Id == id))
                .Where(n => n != null)
                .ToList();

            cycles.Add(cycleNodes!);
            return;
        }

        if (visited.Contains(nodeId))
        {
            return;
        }

        visited.Add(nodeId);
        path.Add(nodeId);

        var dependencies = GetDependencies(nodeId);
        foreach (var dependency in dependencies)
        {
            DetectCyclesRecursive(dependency.Id, visited, path, cycles);
        }

        path.RemoveAt(path.Count - 1);
    }
}
