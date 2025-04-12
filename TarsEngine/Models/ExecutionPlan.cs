using System;
using System.Collections.Generic;
using System.Linq;

namespace TarsEngine.Models;

/// <summary>
/// Represents a plan for executing an improvement
/// </summary>
public class ExecutionPlan
{
    /// <summary>
    /// Gets or sets the ID of the execution plan
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the execution plan
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the execution plan
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ID of the improvement associated with the execution plan
    /// </summary>
    public string ImprovementId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ID of the metascript associated with the execution plan
    /// </summary>
    public string MetascriptId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the steps in the execution plan
    /// </summary>
    public List<ExecutionStep> Steps { get; set; } = new();

    /// <summary>
    /// Gets or sets the timestamp when the execution plan was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the execution plan was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the execution plan was last executed
    /// </summary>
    public DateTime? ExecutedAt { get; set; }

    /// <summary>
    /// Gets or sets the status of the execution plan
    /// </summary>
    public ExecutionPlanStatus Status { get; set; } = ExecutionPlanStatus.Created;

    /// <summary>
    /// Gets or sets the result of the execution plan
    /// </summary>
    public ExecutionPlanResult? Result { get; set; }

    /// <summary>
    /// Gets or sets the execution context
    /// </summary>
    public ExecutionContext? Context { get; set; }

    /// <summary>
    /// Gets or sets additional metadata about the execution plan
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets the total number of steps in the execution plan
    /// </summary>
    public int TotalSteps => Steps.Count;

    /// <summary>
    /// Gets the number of completed steps in the execution plan
    /// </summary>
    public int CompletedSteps => Steps.Count(s => s.Status == ExecutionStepStatus.Completed);

    /// <summary>
    /// Gets the number of failed steps in the execution plan
    /// </summary>
    public int FailedSteps => Steps.Count(s => s.Status == ExecutionStepStatus.Failed);

    /// <summary>
    /// Gets the progress of the execution plan (0.0 to 1.0)
    /// </summary>
    public double Progress => TotalSteps > 0 ? (double)CompletedSteps / TotalSteps : 0.0;

    /// <summary>
    /// Gets whether the execution plan is completed
    /// </summary>
    public bool IsCompleted => Status == ExecutionPlanStatus.Completed || Status == ExecutionPlanStatus.Failed;

    /// <summary>
    /// Gets whether the execution plan is successful
    /// </summary>
    public bool IsSuccessful => Status == ExecutionPlanStatus.Completed && (Result?.IsSuccessful ?? false);

    /// <summary>
    /// Gets the next step to execute
    /// </summary>
    public ExecutionStep? GetNextStep()
    {
        return Steps.FirstOrDefault(s => s.Status == ExecutionStepStatus.Pending);
    }

    /// <summary>
    /// Gets the current step being executed
    /// </summary>
    public ExecutionStep? GetCurrentStep()
    {
        return Steps.FirstOrDefault(s => s.Status == ExecutionStepStatus.InProgress);
    }

    /// <summary>
    /// Gets the dependencies for a step
    /// </summary>
    /// <param name="stepId">The step ID</param>
    /// <returns>The list of dependencies</returns>
    public List<ExecutionStep> GetDependencies(string stepId)
    {
        var step = Steps.FirstOrDefault(s => s.Id == stepId);
        if (step == null)
        {
            return new List<ExecutionStep>();
        }

        return Steps.Where(s => step.Dependencies.Contains(s.Id)).ToList();
    }

    /// <summary>
    /// Gets the dependents for a step
    /// </summary>
    /// <param name="stepId">The step ID</param>
    /// <returns>The list of dependents</returns>
    public List<ExecutionStep> GetDependents(string stepId)
    {
        return Steps.Where(s => s.Dependencies.Contains(stepId)).ToList();
    }

    /// <summary>
    /// Validates the execution plan
    /// </summary>
    /// <returns>True if the execution plan is valid, false otherwise</returns>
    public bool Validate()
    {
        // Check if there are any steps
        if (Steps.Count == 0)
        {
            return false;
        }

        // Check if all dependencies exist
        foreach (var step in Steps)
        {
            foreach (var dependencyId in step.Dependencies)
            {
                if (!Steps.Any(s => s.Id == dependencyId))
                {
                    return false;
                }
            }
        }

        // Check for cycles
        var visited = new HashSet<string>();
        var path = new HashSet<string>();

        foreach (var step in Steps)
        {
            if (!visited.Contains(step.Id))
            {
                if (HasCycle(step.Id, visited, path))
                {
                    return false;
                }
            }
        }

        return true;
    }

    private bool HasCycle(string stepId, HashSet<string> visited, HashSet<string> path)
    {
        visited.Add(stepId);
        path.Add(stepId);

        var step = Steps.FirstOrDefault(s => s.Id == stepId);
        if (step != null)
        {
            foreach (var dependencyId in step.Dependencies)
            {
                if (!visited.Contains(dependencyId))
                {
                    if (HasCycle(dependencyId, visited, path))
                    {
                        return true;
                    }
                }
                else if (path.Contains(dependencyId))
                {
                    return true;
                }
            }
        }

        path.Remove(stepId);
        return false;
    }
}
