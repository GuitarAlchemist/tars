namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a novelty metric
/// </summary>
public class NoveltyMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="NoveltyMetric"/> class
    /// </summary>
    public NoveltyMetric()
    {
        Category = MetricCategory.Novelty;
    }
    
    /// <summary>
    /// Gets or sets the novelty type
    /// </summary>
    public NoveltyType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the reference
    /// </summary>
    public string Reference { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the similarity score
    /// </summary>
    public double SimilarityScore { get; set; }
    
    /// <summary>
    /// Gets the novelty score
    /// </summary>
    public double NoveltyScore => 1.0 - SimilarityScore;
    
    /// <summary>
    /// Gets or sets the novelty dimension
    /// </summary>
    public string Dimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the novelty threshold
    /// </summary>
    public double NoveltyThreshold { get; set; } = 0.5;
    
    /// <summary>
    /// Gets whether the novelty exceeds the threshold
    /// </summary>
    public bool ExceedsThreshold => NoveltyScore > NoveltyThreshold;
}

/// <summary>
/// Represents a novelty type
/// </summary>
public enum NoveltyType
{
    /// <summary>
    /// Content novelty
    /// </summary>
    Content,
    
    /// <summary>
    /// Structure novelty
    /// </summary>
    Structure,
    
    /// <summary>
    /// Approach novelty
    /// </summary>
    Approach,
    
    /// <summary>
    /// Concept novelty
    /// </summary>
    Concept,
    
    /// <summary>
    /// Solution novelty
    /// </summary>
    Solution,
    
    /// <summary>
    /// Other novelty
    /// </summary>
    Other
}
