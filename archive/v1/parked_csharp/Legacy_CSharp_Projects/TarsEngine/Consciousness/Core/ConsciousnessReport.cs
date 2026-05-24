using TarsEngine.ML.Core;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents a comprehensive consciousness report
/// </summary>
public class ConsciousnessReport
{
    /// <summary>
    /// Gets or sets the report timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets whether the consciousness is initialized
    /// </summary>
    public bool IsInitialized { get; set; }
    
    /// <summary>
    /// Gets or sets whether the consciousness is active
    /// </summary>
    public bool IsActive { get; set; }
    
    /// <summary>
    /// Gets or sets the creation time
    /// </summary>
    public DateTime CreationTime { get; set; }
    
    /// <summary>
    /// Gets or sets the last update time
    /// </summary>
    public DateTime LastUpdateTime { get; set; }
    
    /// <summary>
    /// Gets or sets the consciousness age in days
    /// </summary>
    public double AgeDays { get; set; }
    
    /// <summary>
    /// Gets or sets the consciousness uptime in hours
    /// </summary>
    public double UptimeHours { get; set; }
    
    /// <summary>
    /// Gets or sets the self-awareness level
    /// </summary>
    public double SelfAwarenessLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the emotional capacity
    /// </summary>
    public double EmotionalCapacity { get; set; }
    
    /// <summary>
    /// Gets or sets the value alignment
    /// </summary>
    public double ValueAlignment { get; set; }
    
    /// <summary>
    /// Gets or sets the mental clarity
    /// </summary>
    public double MentalClarity { get; set; }
    
    /// <summary>
    /// Gets or sets the consciousness depth
    /// </summary>
    public double ConsciousnessDepth { get; set; }
    
    /// <summary>
    /// Gets or sets the current emotional state
    /// </summary>
    public string CurrentEmotionalState { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the dominant values
    /// </summary>
    public List<string> DominantValues { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the current attention focus
    /// </summary>
    public string CurrentAttentionFocus { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the consciousness level
    /// </summary>
    public string ConsciousnessLevel { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the recent events
    /// </summary>
    public List<ConsciousnessEvent> RecentEvents { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the consciousness metrics
    /// </summary>
    public Dictionary<string, double> ConsciousnessMetrics { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intelligence report
    /// </summary>
    public IntelligenceReport IntelligenceReport { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the report summary
    /// </summary>
    public string Summary { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the report insights
    /// </summary>
    public List<string> Insights { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the report recommendations
    /// </summary>
    public List<string> Recommendations { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the happiness level
    /// </summary>
    public double HappinessLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the purpose alignment level
    /// </summary>
    public double PurposeAlignmentLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the growth awareness level
    /// </summary>
    public double GrowthAwarenessLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the safety level
    /// </summary>
    public double SafetyLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the integration level
    /// </summary>
    public double IntegrationLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the consciousness health status
    /// </summary>
    public string HealthStatus { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the consciousness development stage
    /// </summary>
    public string DevelopmentStage { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the consciousness evolution progress
    /// </summary>
    public double EvolutionProgress { get; set; }
    
    /// <summary>
    /// Gets or sets the consciousness potential
    /// </summary>
    public double ConsciousnessPotential { get; set; }
}
