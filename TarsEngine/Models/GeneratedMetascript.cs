using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents a generated metascript
/// </summary>
public class GeneratedMetascript
{
    /// <summary>
    /// Gets or sets the ID of the metascript
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the metascript
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the metascript
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the metascript code
    /// </summary>
    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the programming language of the metascript
    /// </summary>
    public string Language { get; set; } = "meta";

    /// <summary>
    /// Gets or sets the template ID used to generate the metascript
    /// </summary>
    public string? TemplateId { get; set; }

    /// <summary>
    /// Gets or sets the pattern ID that triggered the metascript generation
    /// </summary>
    public string? PatternId { get; set; }

    /// <summary>
    /// Gets or sets the parameters used to generate the metascript
    /// </summary>
    public Dictionary<string, string> Parameters { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Gets or sets the timestamp when the metascript was generated
    /// </summary>
    public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the metascript was last executed
    /// </summary>
    public DateTime? LastExecutedAt { get; set; }

    /// <summary>
    /// Gets or sets the execution status of the metascript
    /// </summary>
    public MetascriptExecutionStatus ExecutionStatus { get; set; } = MetascriptExecutionStatus.NotExecuted;

    /// <summary>
    /// Gets or sets the execution result of the metascript
    /// </summary>
    public string? ExecutionResult { get; set; }

    /// <summary>
    /// Gets or sets the execution time of the metascript in milliseconds
    /// </summary>
    public long? ExecutionTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the validation status of the metascript
    /// </summary>
    public MetascriptValidationStatus ValidationStatus { get; set; } = MetascriptValidationStatus.NotValidated;

    /// <summary>
    /// Gets or sets the validation messages for the metascript
    /// </summary>
    public List<string> ValidationMessages { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the expected improvement from applying the metascript
    /// </summary>
    public string? ExpectedImprovement { get; set; }

    /// <summary>
    /// Gets or sets the impact score of applying the metascript (0.0 to 1.0)
    /// </summary>
    public double ImpactScore { get; set; }

    /// <summary>
    /// Gets or sets the difficulty score of applying the metascript (0.0 to 1.0)
    /// </summary>
    public double DifficultyScore { get; set; }

    /// <summary>
    /// Gets or sets the priority score of the metascript (0.0 to 1.0)
    /// </summary>
    public double PriorityScore { get; set; }

    /// <summary>
    /// Gets or sets the list of files affected by the metascript
    /// </summary>
    public List<string> AffectedFiles { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of tags associated with the metascript
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets additional metadata about the metascript
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
