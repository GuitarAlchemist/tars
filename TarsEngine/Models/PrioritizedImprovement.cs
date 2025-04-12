using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents a prioritized improvement
/// </summary>
public class PrioritizedImprovement
{
    /// <summary>
    /// Gets or sets the ID of the improvement
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the improvement
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the improvement
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the category of the improvement
    /// </summary>
    public ImprovementCategory Category { get; set; } = ImprovementCategory.Other;

    /// <summary>
    /// Gets or sets the impact of the improvement
    /// </summary>
    public ImprovementImpact Impact { get; set; } = ImprovementImpact.Medium;

    /// <summary>
    /// Gets or sets the effort required for the improvement
    /// </summary>
    public ImprovementEffort Effort { get; set; } = ImprovementEffort.Medium;

    /// <summary>
    /// Gets or sets the risk of the improvement
    /// </summary>
    public ImprovementRisk Risk { get; set; } = ImprovementRisk.Medium;

    /// <summary>
    /// Gets or sets the impact score of the improvement (0.0 to 1.0)
    /// </summary>
    public double ImpactScore { get; set; }

    /// <summary>
    /// Gets or sets the effort score of the improvement (0.0 to 1.0)
    /// </summary>
    public double EffortScore { get; set; }

    /// <summary>
    /// Gets or sets the risk score of the improvement (0.0 to 1.0)
    /// </summary>
    public double RiskScore { get; set; }

    /// <summary>
    /// Gets or sets the alignment score of the improvement (0.0 to 1.0)
    /// </summary>
    public double AlignmentScore { get; set; }

    /// <summary>
    /// Gets or sets the priority score of the improvement (0.0 to 1.0)
    /// </summary>
    public double PriorityScore { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the improvement was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the improvement was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the improvement was last prioritized
    /// </summary>
    public DateTime? PrioritizedAt { get; set; }

    /// <summary>
    /// Gets or sets the status of the improvement
    /// </summary>
    public ImprovementStatus Status { get; set; } = ImprovementStatus.Pending;

    /// <summary>
    /// Gets or sets the ID of the metascript associated with the improvement
    /// </summary>
    public string? MetascriptId { get; set; }

    /// <summary>
    /// Gets or sets the ID of the pattern match associated with the improvement
    /// </summary>
    public string? PatternMatchId { get; set; }

    /// <summary>
    /// Gets or sets the list of files affected by the improvement
    /// </summary>
    public List<string> AffectedFiles { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of dependencies for the improvement
    /// </summary>
    public List<string> Dependencies { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of dependents for the improvement
    /// </summary>
    public List<string> Dependents { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of strategic goals aligned with the improvement
    /// </summary>
    public List<string> StrategicGoals { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of tags associated with the improvement
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the priority rank of the improvement (1-10, with 1 being highest priority)
    /// </summary>
    public int PriorityRank { get; set; } = 5;

    /// <summary>
    /// Gets or sets the estimated duration in minutes for implementing the improvement
    /// </summary>
    public int EstimatedDurationMinutes { get; set; } = 60;

    /// <summary>
    /// Gets or sets the actual duration in minutes for implementing the improvement
    /// </summary>
    public int? ActualDurationMinutes { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the improvement was completed
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets additional metadata about the improvement
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
