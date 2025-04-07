using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents a generated improvement
/// </summary>
public class GeneratedImprovement
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
    /// Gets or sets the timestamp when the improvement was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the improvement was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

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
    public List<string> AffectedFiles { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of tags associated with the improvement
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets additional metadata about the improvement
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
