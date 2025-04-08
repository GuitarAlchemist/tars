using System;
using System.Collections.Generic;

namespace TarsEngine.ML.Core;

/// <summary>
/// Represents a projection of future intelligence growth
/// </summary>
public class IntelligenceProjection
{
    /// <summary>
    /// Gets or sets the projection ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the creation timestamp
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the current intelligence score
    /// </summary>
    public double CurrentIntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic current intelligence score
    /// </summary>
    public double LogCurrentIntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the current growth rate
    /// </summary>
    public double CurrentGrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic current growth rate
    /// </summary>
    public double LogCurrentGrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the projected scores at future time points
    /// </summary>
    public Dictionary<TimeSpan, double> ProjectedScores { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the logarithmic projected scores
    /// </summary>
    public Dictionary<TimeSpan, double> LogProjectedScores { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the projected milestones
    /// </summary>
    public List<ProjectedMilestone> ProjectedMilestones { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the growth model used
    /// </summary>
    public string GrowthModel { get; set; } = "Linear";
    
    /// <summary>
    /// Gets or sets the confidence level (0.0-1.0)
    /// </summary>
    public double ConfidenceLevel { get; set; } = 0.8;
    
    /// <summary>
    /// Gets or sets the projection description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the projection assumptions
    /// </summary>
    public List<string> Assumptions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the projection risks
    /// </summary>
    public List<string> Risks { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the projection opportunities
    /// </summary>
    public List<string> Opportunities { get; set; } = new();
}

/// <summary>
/// Represents a projected milestone
/// </summary>
public class ProjectedMilestone
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
    /// Gets or sets the projected achievement date
    /// </summary>
    public DateTime ProjectedAchievementDate { get; set; }
    
    /// <summary>
    /// Gets or sets the confidence level (0.0-1.0)
    /// </summary>
    public double ConfidenceLevel { get; set; } = 0.8;
    
    /// <summary>
    /// Gets or sets the key capabilities gained
    /// </summary>
    public List<string> KeyCapabilitiesGained { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the significance
    /// </summary>
    public string Significance { get; set; } = string.Empty;
}
