namespace TarsEngine.Models;

/// <summary>
/// Represents a code pattern that can be matched in source code
/// </summary>
public class CodePattern
{
    /// <summary>
    /// Gets or sets the ID of the pattern
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the pattern
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the pattern
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the programming language the pattern applies to
    /// </summary>
    public string Language { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the pattern definition
    /// </summary>
    public string Pattern { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the pattern language used (e.g., Regex, AST, Custom)
    /// </summary>
    public string PatternLanguage { get; set; } = "Regex";

    /// <summary>
    /// Gets or sets the suggested replacement for matched code
    /// </summary>
    public string? Replacement { get; set; }

    /// <summary>
    /// Gets or sets the explanation for the suggested replacement
    /// </summary>
    public string? ReplacementExplanation { get; set; }

    /// <summary>
    /// Gets or sets the expected improvement from applying the replacement
    /// </summary>
    public string? ExpectedImprovement { get; set; }

    /// <summary>
    /// Gets or sets the severity of the pattern (0-4, with 4 being most severe)
    /// </summary>
    public int Severity { get; set; }

    /// <summary>
    /// Gets or sets the confidence threshold for matching (0.0 to 1.0)
    /// </summary>
    public double ConfidenceThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the impact score of applying the replacement (0.0 to 1.0)
    /// </summary>
    public double ImpactScore { get; set; }

    /// <summary>
    /// Gets or sets the difficulty score of applying the replacement (0.0 to 1.0)
    /// </summary>
    public double DifficultyScore { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the pattern was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the pattern was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the pattern was last used
    /// </summary>
    public DateTime? LastUsedAt { get; set; }

    /// <summary>
    /// Gets or sets the number of times the pattern has been used
    /// </summary>
    public int UsageCount { get; set; }

    /// <summary>
    /// Gets or sets the number of successful matches
    /// </summary>
    public int SuccessCount { get; set; }

    /// <summary>
    /// Gets or sets the number of false positives
    /// </summary>
    public int FalsePositiveCount { get; set; }

    /// <summary>
    /// Gets or sets the list of example code snippets that match the pattern
    /// </summary>
    public List<string> Examples { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of tags associated with the pattern
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata about the pattern
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the pattern-specific options
    /// </summary>
    public Dictionary<string, string> Options { get; set; } = new();

    /// <summary>
    /// Gets or sets the related patterns
    /// </summary>
    public List<string> RelatedPatterns { get; set; } = new();
}
