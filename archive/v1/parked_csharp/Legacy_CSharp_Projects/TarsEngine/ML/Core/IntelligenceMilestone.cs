namespace TarsEngine.ML.Core;

/// <summary>
/// Represents a milestone in intelligence development
/// </summary>
public class IntelligenceMilestone
{
    /// <summary>
    /// Gets or sets the milestone ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the milestone name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the milestone description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target intelligence score
    /// </summary>
    public double TargetIntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic target intelligence score
    /// </summary>
    public double LogTargetIntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the target intelligence ratio
    /// </summary>
    public double TargetIntelligenceRatio { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic target intelligence ratio
    /// </summary>
    public double LogTargetIntelligenceRatio { get; set; }
    
    /// <summary>
    /// Gets or sets whether the milestone has been achieved
    /// </summary>
    public bool IsAchieved { get; set; }
    
    /// <summary>
    /// Gets or sets the achievement date
    /// </summary>
    public DateTime? AchievedAt { get; set; }
    
    /// <summary>
    /// Gets or sets the estimated achievement date
    /// </summary>
    public DateTime? EstimatedAchievementDate { get; set; }
    
    /// <summary>
    /// Gets or sets the key capabilities gained
    /// </summary>
    public List<string> KeyCapabilitiesGained { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the significance
    /// </summary>
    public string Significance { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the prerequisites
    /// </summary>
    public List<string> Prerequisites { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the next milestones
    /// </summary>
    public List<string> NextMilestones { get; set; } = new();
}
