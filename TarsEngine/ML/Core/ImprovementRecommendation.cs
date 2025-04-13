namespace TarsEngine.ML.Core;

/// <summary>
/// Represents a recommendation for intelligence improvement
/// </summary>
public class ImprovementRecommendation
{
    /// <summary>
    /// Gets or sets the recommendation ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the target dimension
    /// </summary>
    public string TargetDimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the current value
    /// </summary>
    public double CurrentValue { get; set; }
    
    /// <summary>
    /// Gets or sets the target value
    /// </summary>
    public double TargetValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic current value
    /// </summary>
    public double LogCurrentValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic target value
    /// </summary>
    public double LogTargetValue { get; set; }
    
    /// <summary>
    /// Gets or sets the priority (1-10, with 10 being highest)
    /// </summary>
    public int Priority { get; set; }
    
    /// <summary>
    /// Gets or sets the recommendation description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the suggested actions
    /// </summary>
    public List<string> SuggestedActions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the expected impact
    /// </summary>
    public string ExpectedImpact { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the estimated effort (1-10, with 10 being highest)
    /// </summary>
    public int EstimatedEffort { get; set; }
    
    /// <summary>
    /// Gets or sets the estimated time to implement in days
    /// </summary>
    public double EstimatedTimeToImplementDays { get; set; }
    
    /// <summary>
    /// Gets or sets the expected growth rate after implementation
    /// </summary>
    public double ExpectedGrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic expected growth rate
    /// </summary>
    public double LogExpectedGrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the related dimensions that will also improve
    /// </summary>
    public List<string> RelatedDimensions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the prerequisites
    /// </summary>
    public List<string> Prerequisites { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the creation timestamp
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the implementation status
    /// </summary>
    public string ImplementationStatus { get; set; } = "Pending";
}
