using System;
using System.Collections.Generic;

namespace TarsEngine.Intelligence.Measurement;

/// <summary>
/// Represents an intelligence progression report
/// </summary>
public class IntelligenceProgressionReport
{
    /// <summary>
    /// Gets or sets the start time
    /// </summary>
    public DateTime StartTime { get; set; }
    
    /// <summary>
    /// Gets or sets the end time
    /// </summary>
    public DateTime EndTime { get; set; }
    
    /// <summary>
    /// Gets or sets the generated at timestamp
    /// </summary>
    public DateTime GeneratedAt { get; set; }
    
    /// <summary>
    /// Gets or sets the overall intelligence score
    /// </summary>
    public double OverallIntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the learning curve analysis
    /// </summary>
    public LearningCurveAnalysis LearningCurveAnalysis { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the modification analysis
    /// </summary>
    public ModificationAnalysis ModificationAnalysis { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the visualization data
    /// </summary>
    public VisualizationData VisualizationData { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the key insights
    /// </summary>
    public List<string> KeyInsights { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the recommendations
    /// </summary>
    public List<string> Recommendations { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the growth areas
    /// </summary>
    public List<string> GrowthAreas { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the strengths
    /// </summary>
    public List<string> Strengths { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intelligence progression milestones
    /// </summary>
    public List<IntelligenceMilestone> Milestones { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intelligence progression forecast
    /// </summary>
    public IntelligenceForecast Forecast { get; set; } = new();
    
    /// <summary>
    /// Gets the time period in days
    /// </summary>
    public int TimePeriodDays => (int)(EndTime - StartTime).TotalDays;
    
    /// <summary>
    /// Gets the intelligence growth rate per day
    /// </summary>
    public double IntelligenceGrowthRatePerDay { get; set; }
    
    /// <summary>
    /// Gets the logarithmic intelligence growth rate per day
    /// </summary>
    public double LogIntelligenceGrowthRatePerDay { get; set; }
    
    /// <summary>
    /// Gets or sets the domain-specific intelligence scores
    /// </summary>
    public Dictionary<string, double> DomainSpecificScores { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the skill-specific intelligence scores
    /// </summary>
    public Dictionary<string, double> SkillSpecificScores { get; set; } = new();
}

/// <summary>
/// Represents an intelligence milestone
/// </summary>
public class IntelligenceMilestone
{
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intelligence score
    /// </summary>
    public double IntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the milestone type
    /// </summary>
    public MilestoneType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the significance level
    /// </summary>
    public int SignificanceLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the domains
    /// </summary>
    public List<string> Domains { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the skills
    /// </summary>
    public List<string> Skills { get; set; } = new();
}

/// <summary>
/// Represents a milestone type
/// </summary>
public enum MilestoneType
{
    /// <summary>
    /// Learning milestone
    /// </summary>
    Learning,
    
    /// <summary>
    /// Problem solving milestone
    /// </summary>
    ProblemSolving,
    
    /// <summary>
    /// Creativity milestone
    /// </summary>
    Creativity,
    
    /// <summary>
    /// Efficiency milestone
    /// </summary>
    Efficiency,
    
    /// <summary>
    /// Adaptability milestone
    /// </summary>
    Adaptability,
    
    /// <summary>
    /// Other milestone
    /// </summary>
    Other
}

/// <summary>
/// Represents an intelligence forecast
/// </summary>
public class IntelligenceForecast
{
    /// <summary>
    /// Gets or sets the forecast period in days
    /// </summary>
    public int ForecastPeriodDays { get; set; } = 30;
    
    /// <summary>
    /// Gets or sets the forecast values
    /// </summary>
    public Dictionary<DateTime, double> ForecastValues { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the forecast confidence intervals
    /// </summary>
    public Dictionary<DateTime, (double Lower, double Upper)> ConfidenceIntervals { get; set; } = 
        new();
    
    /// <summary>
    /// Gets or sets the expected milestones
    /// </summary>
    public List<IntelligenceMilestone> ExpectedMilestones { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the forecast model type
    /// </summary>
    public string ForecastModelType { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the forecast accuracy
    /// </summary>
    public double ForecastAccuracy { get; set; }
    
    /// <summary>
    /// Gets or sets the forecast confidence level
    /// </summary>
    public double ConfidenceLevel { get; set; } = 0.95;
}
