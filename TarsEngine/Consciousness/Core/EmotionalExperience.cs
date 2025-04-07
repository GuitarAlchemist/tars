using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an emotional experience
/// </summary>
public class EmotionalExperience
{
    /// <summary>
    /// Gets or sets the experience ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the emotion name
    /// </summary>
    public string EmotionName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the trigger
    /// </summary>
    public string Trigger { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intensity (0.0 to 1.0)
    /// </summary>
    public double Intensity { get; set; }
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the emotional capacity at the time of the experience
    /// </summary>
    public double EmotionalCapacity { get; set; }
    
    /// <summary>
    /// Gets or sets the emotional intelligence at the time of the experience
    /// </summary>
    public double EmotionalIntelligence { get; set; }
    
    /// <summary>
    /// Gets or sets the experience tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the experience context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the related experience IDs
    /// </summary>
    public List<string> RelatedExperienceIds { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; }
    
    /// <summary>
    /// Gets or sets the response
    /// </summary>
    public string Response { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the impact (0.0 to 1.0)
    /// </summary>
    public double Impact { get; set; }
    
    /// <summary>
    /// Gets or sets the learning value (0.0 to 1.0)
    /// </summary>
    public double LearningValue { get; set; }
    
    /// <summary>
    /// Gets or sets the memory strength (0.0 to 1.0)
    /// </summary>
    public double MemoryStrength { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the last recalled timestamp
    /// </summary>
    public DateTime LastRecalledTimestamp { get; set; } = DateTime.MinValue;
    
    /// <summary>
    /// Gets or sets the recall count
    /// </summary>
    public int RecallCount { get; set; } = 0;
    
    /// <summary>
    /// Records a recall of this experience
    /// </summary>
    public void RecordRecall()
    {
        RecallCount++;
        LastRecalledTimestamp = DateTime.UtcNow;
        
        // Strengthen memory with each recall (up to a point)
        MemoryStrength = Math.Min(1.0, MemoryStrength + 0.05);
    }
}
