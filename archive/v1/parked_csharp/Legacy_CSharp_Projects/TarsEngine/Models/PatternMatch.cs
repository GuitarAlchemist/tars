namespace TarsEngine.Models;

/// <summary>
/// Represents a match of a code pattern in source code
/// </summary>
public class PatternMatch
{
    /// <summary>
    /// Gets or sets the ID of the match
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the ID of the pattern that was matched
    /// </summary>
    public string PatternId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the pattern that was matched
    /// </summary>
    public string PatternName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the path to the file where the pattern was found
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the programming language of the code
    /// </summary>
    public string Language { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location of the match in the code
    /// </summary>
    public CodeLocation Location { get; set; } = new();

    /// <summary>
    /// Gets or sets the matched text
    /// </summary>
    public string MatchedText { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the context around the match
    /// </summary>
    public string Context { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score of the match (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the timestamp when the match was found
    /// </summary>
    public DateTime MatchedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the suggested replacement for the matched text
    /// </summary>
    public string? SuggestedReplacement { get; set; }

    /// <summary>
    /// Gets or sets the explanation for the suggested replacement
    /// </summary>
    public string? ReplacementExplanation { get; set; }

    /// <summary>
    /// Gets or sets the expected improvement from applying the replacement
    /// </summary>
    public string? ExpectedImprovement { get; set; }

    /// <summary>
    /// Gets or sets the impact score of applying the replacement (0.0 to 1.0)
    /// </summary>
    public double ImpactScore { get; set; }

    /// <summary>
    /// Gets or sets the difficulty score of applying the replacement (0.0 to 1.0)
    /// </summary>
    public double DifficultyScore { get; set; }

    /// <summary>
    /// Gets or sets the list of tags associated with the match
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets additional metadata about the match
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
