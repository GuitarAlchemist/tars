using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an emotional association with a trigger
/// </summary>
public class EmotionalAssociation
{
    /// <summary>
    /// Gets or sets the trigger
    /// </summary>
    public string Trigger { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the emotion associations (emotion name to intensity)
    /// </summary>
    public Dictionary<string, double> EmotionAssociations { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the last experienced timestamp
    /// </summary>
    public DateTime LastExperienced { get; set; } = DateTime.MinValue;
    
    /// <summary>
    /// Gets or sets the experience count
    /// </summary>
    public int ExperienceCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the association strength (0.0 to 1.0)
    /// </summary>
    public double AssociationStrength { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the association context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the association tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the association category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the association description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets the dominant emotion
    /// </summary>
    public string DominantEmotion
    {
        get
        {
            string dominantEmotion = string.Empty;
            double maxIntensity = 0.0;
            
            foreach (var (emotion, intensity) in EmotionAssociations)
            {
                if (intensity > maxIntensity)
                {
                    maxIntensity = intensity;
                    dominantEmotion = emotion;
                }
            }
            
            return dominantEmotion;
        }
    }
    
    /// <summary>
    /// Gets the dominant emotion intensity
    /// </summary>
    public double DominantEmotionIntensity
    {
        get
        {
            double maxIntensity = 0.0;
            
            foreach (var (_, intensity) in EmotionAssociations)
            {
                if (intensity > maxIntensity)
                {
                    maxIntensity = intensity;
                }
            }
            
            return maxIntensity;
        }
    }
    
    /// <summary>
    /// Updates the association with a new emotional experience
    /// </summary>
    /// <param name="emotionName">The emotion name</param>
    /// <param name="intensity">The intensity</param>
    public void UpdateAssociation(string emotionName, double intensity)
    {
        // Update or add emotion association
        if (EmotionAssociations.TryGetValue(emotionName, out var existingIntensity))
        {
            // Weighted average (giving more weight to existing association for stability)
            double weight = Math.Min(0.8, (double)ExperienceCount / (ExperienceCount + 5));
            EmotionAssociations[emotionName] = (existingIntensity * weight) + (intensity * (1.0 - weight));
        }
        else
        {
            EmotionAssociations[emotionName] = intensity;
        }
        
        // Update metadata
        LastExperienced = DateTime.UtcNow;
        ExperienceCount++;
        
        // Increase association strength with repeated experiences
        AssociationStrength = Math.Min(1.0, AssociationStrength + (0.05 / Math.Sqrt(ExperienceCount)));
    }
}
