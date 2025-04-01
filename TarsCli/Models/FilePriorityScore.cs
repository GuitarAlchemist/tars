namespace TarsCli.Models;

/// <summary>
/// Represents a priority score for a file in the auto-improvement process
/// </summary>
public class FilePriorityScore
{
    /// <summary>
    /// The path of the file
    /// </summary>
    public string FilePath { get; set; }

    /// <summary>
    /// The base score of the file (higher is higher priority)
    /// </summary>
    public double BaseScore { get; set; }

    /// <summary>
    /// The content score of the file (based on content analysis)
    /// </summary>
    public double ContentScore { get; set; }

    /// <summary>
    /// The recency score of the file (based on last modified time)
    /// </summary>
    public double RecencyScore { get; set; }

    /// <summary>
    /// The complexity score of the file (based on content complexity)
    /// </summary>
    public double ComplexityScore { get; set; }

    /// <summary>
    /// The improvement potential score of the file (based on previous analysis)
    /// </summary>
    public double ImprovementPotentialScore { get; set; }

    /// <summary>
    /// The total score of the file (sum of all scores)
    /// </summary>
    public double TotalScore => BaseScore + ContentScore + RecencyScore + ComplexityScore + ImprovementPotentialScore;

    /// <summary>
    /// The factors that contributed to the score
    /// </summary>
    public Dictionary<string, double> ScoreFactors { get; } = new Dictionary<string, double>();

    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="filePath">The path of the file</param>
    public FilePriorityScore(string filePath)
    {
        FilePath = filePath;
    }

    /// <summary>
    /// Add a score factor
    /// </summary>
    /// <param name="factorName">The name of the factor</param>
    /// <param name="score">The score value</param>
    public void AddFactor(string factorName, double score)
    {
        ScoreFactors[factorName] = score;
    }

    /// <summary>
    /// Get a description of the score
    /// </summary>
    /// <returns>A string describing the score</returns>
    public string GetDescription()
    {
        var description = $"Total Score: {TotalScore:F2}\n";
        description += "Score Factors:\n";
            
        foreach (var factor in ScoreFactors)
        {
            description += $"- {factor.Key}: {factor.Value:F2}\n";
        }
            
        return description;
    }
}