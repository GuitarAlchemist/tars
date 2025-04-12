using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents an autobiographical memory entry
/// </summary>
public class MemoryEntry
{
    /// <summary>
    /// Gets or sets the memory ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the memory category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the memory content
    /// </summary>
    public string Content { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the memory timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the memory importance (0.0 to 1.0)
    /// </summary>
    public double Importance { get; set; }
    
    /// <summary>
    /// Gets or sets the self-awareness level at the time of the memory
    /// </summary>
    public double SelfAwarenessLevel { get; set; }
    
    /// <summary>
    /// Gets or sets the memory tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the memory context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the emotional association
    /// </summary>
    public string EmotionalAssociation { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the emotional intensity (0.0 to 1.0)
    /// </summary>
    public double EmotionalIntensity { get; set; }
    
    /// <summary>
    /// Gets or sets the memory accessibility (0.0 to 1.0)
    /// </summary>
    public double Accessibility { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the memory vividness (0.0 to 1.0)
    /// </summary>
    public double Vividness { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the memory accuracy (0.0 to 1.0)
    /// </summary>
    public double Accuracy { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the related memory IDs
    /// </summary>
    public List<string> RelatedMemoryIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the last accessed timestamp
    /// </summary>
    public DateTime LastAccessedTimestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the access count
    /// </summary>
    public int AccessCount { get; set; } = 0;
    
    /// <summary>
    /// Records an access to this memory
    /// </summary>
    public void RecordAccess()
    {
        AccessCount++;
        LastAccessedTimestamp = DateTime.UtcNow;
    }
}
