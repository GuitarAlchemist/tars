using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents an information gap
/// </summary>
public class InformationGap
{
    /// <summary>
    /// Gets or sets the gap ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the question
    /// </summary>
    public string Question { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the importance (0.0 to 1.0)
    /// </summary>
    public double Importance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the gap size (0.0 to 1.0)
    /// </summary>
    public double GapSize { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the last explored timestamp
    /// </summary>
    public DateTime? LastExploredTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the exploration count
    /// </summary>
    public int ExplorationCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the exploration IDs
    /// </summary>
    public List<string> ExplorationIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the gap tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the gap context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the gap status
    /// </summary>
    public GapStatus Status { get; set; } = GapStatus.Identified;
    
    /// <summary>
    /// Gets or sets the gap priority
    /// </summary>
    public int Priority { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the related gap IDs
    /// </summary>
    public List<string> RelatedGapIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the gap notes
    /// </summary>
    public string Notes { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets the gap priority score
    /// </summary>
    public double PriorityScore => Importance * GapSize;
    
    /// <summary>
    /// Adds an exploration
    /// </summary>
    /// <param name="explorationId">The exploration ID</param>
    public void AddExploration(string explorationId)
    {
        ExplorationIds.Add(explorationId);
        ExplorationCount++;
        LastExploredTimestamp = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Updates the gap size
    /// </summary>
    /// <param name="newSize">The new size</param>
    public void UpdateGapSize(double newSize)
    {
        GapSize = Math.Max(0.0, Math.Min(1.0, newSize));
        
        // Update status based on gap size
        if (GapSize < 0.2)
        {
            Status = GapStatus.Filled;
        }
        else if (GapSize < 0.6)
        {
            Status = GapStatus.PartiallyFilled;
        }
        else
        {
            Status = GapStatus.Identified;
        }
    }
    
    /// <summary>
    /// Adds a related gap
    /// </summary>
    /// <param name="gapId">The gap ID</param>
    public void AddRelatedGap(string gapId)
    {
        if (!RelatedGapIds.Contains(gapId))
        {
            RelatedGapIds.Add(gapId);
        }
    }
    
    /// <summary>
    /// Updates the importance
    /// </summary>
    /// <param name="newImportance">The new importance</param>
    public void UpdateImportance(double newImportance)
    {
        Importance = Math.Max(0.0, Math.Min(1.0, newImportance));
    }
}
