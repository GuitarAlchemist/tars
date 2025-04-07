using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an emotion
/// </summary>
public class Emotion
{
    private double _currentIntensity;
    private DateTime _lastIntensityChangeTime = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the emotion name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the emotion category
    /// </summary>
    public EmotionCategory Category { get; set; }
    
    /// <summary>
    /// Gets or sets the current intensity (0.0 to 1.0)
    /// </summary>
    public double CurrentIntensity
    {
        get => _currentIntensity;
        set
        {
            if (Math.Abs(_currentIntensity - value) > 0.01)
            {
                _lastIntensityChangeTime = DateTime.UtcNow;
            }
            _currentIntensity = value;
        }
    }
    
    /// <summary>
    /// Gets or sets the maximum intensity (0.0 to 1.0)
    /// </summary>
    public double MaxIntensity { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the decay rate (how quickly the emotion fades)
    /// </summary>
    public double DecayRate { get; set; } = 0.01;
    
    /// <summary>
    /// Gets or sets the related emotions
    /// </summary>
    public List<string> RelatedEmotions { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the emotion description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the emotion tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the emotion triggers
    /// </summary>
    public List<string> Triggers { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the emotion expressions
    /// </summary>
    public List<string> Expressions { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the emotion action tendencies
    /// </summary>
    public List<string> ActionTendencies { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets the sustained duration (how long the emotion has been at current intensity)
    /// </summary>
    public TimeSpan SustainedDuration => DateTime.UtcNow - _lastIntensityChangeTime;
    
    /// <summary>
    /// Gets or sets the last experienced timestamp
    /// </summary>
    public DateTime LastExperienced { get; set; } = DateTime.MinValue;
    
    /// <summary>
    /// Gets or sets the experience count
    /// </summary>
    public int ExperienceCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the average intensity
    /// </summary>
    public double AverageIntensity { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the maximum experienced intensity
    /// </summary>
    public double MaxExperiencedIntensity { get; set; } = 0.0;
    
    /// <summary>
    /// Records an experience of this emotion
    /// </summary>
    /// <param name="intensity">The intensity</param>
    public void RecordExperience(double intensity)
    {
        LastExperienced = DateTime.UtcNow;
        ExperienceCount++;
        
        // Update average intensity
        AverageIntensity = ((AverageIntensity * (ExperienceCount - 1)) + intensity) / ExperienceCount;
        
        // Update maximum experienced intensity
        MaxExperiencedIntensity = Math.Max(MaxExperiencedIntensity, intensity);
    }
}
