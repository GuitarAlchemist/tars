using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents an insight (legacy version)
/// </summary>
public class InsightLegacy
{
    /// <summary>
    /// Gets or sets the insight ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the insight description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight generation method
    /// </summary>
    public InsightGenerationMethod Method { get; set; } = InsightGenerationMethod.ConnectionDiscovery;
    
    /// <summary>
    /// Gets or sets the insight significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the insight timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the insight context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the insight tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the insight implications
    /// </summary>
    public List<string> Implications { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the insight new perspective
    /// </summary>
    public string NewPerspective { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight breakthrough
    /// </summary>
    public string Breakthrough { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight synthesis
    /// </summary>
    public string Synthesis { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight application
    /// </summary>
    public string Application { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight verification status
    /// </summary>
    public VerificationStatus VerificationStatus { get; set; } = VerificationStatus.Unverified;
    
    /// <summary>
    /// Gets or sets the insight verification notes
    /// </summary>
    public string VerificationNotes { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insight impact (0.0 to 1.0)
    /// </summary>
    public double Impact { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the insight application areas
    /// </summary>
    public List<string> ApplicationAreas { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the insight related insight IDs
    /// </summary>
    public List<string> RelatedInsightIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the insight source
    /// </summary>
    public string Source { get; set; } = "InsightGeneration";
    
    /// <summary>
    /// Verifies the insight
    /// </summary>
    /// <param name="isVerified">Whether the insight is verified</param>
    /// <param name="notes">The verification notes</param>
    public void Verify(bool isVerified, string notes)
    {
        VerificationStatus = isVerified ? VerificationStatus.Verified : VerificationStatus.Falsified;
        VerificationNotes = notes;
    }
    
    /// <summary>
    /// Records an impact for the insight
    /// </summary>
    /// <param name="impact">The impact</param>
    public void RecordImpact(double impact)
    {
        Impact = Math.Max(0.0, Math.Min(1.0, impact));
    }
    
    /// <summary>
    /// Adds an application area
    /// </summary>
    /// <param name="applicationArea">The application area</param>
    public void AddApplicationArea(string applicationArea)
    {
        ApplicationAreas.Add(applicationArea);
    }
    
    /// <summary>
    /// Adds a related insight
    /// </summary>
    /// <param name="insightId">The insight ID</param>
    public void AddRelatedInsight(string insightId)
    {
        if (!RelatedInsightIds.Contains(insightId))
        {
            RelatedInsightIds.Add(insightId);
        }
    }
    
    /// <summary>
    /// Adds an implication
    /// </summary>
    /// <param name="implication">The implication</param>
    public void AddImplication(string implication)
    {
        Implications.Add(implication);
    }
}
