namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a concept node in the semantic network
/// </summary>
public class ConceptNode
{
    /// <summary>
    /// Gets or sets the node ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the node name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the node attributes
    /// </summary>
    public Dictionary<string, double> Attributes { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the connection IDs
    /// </summary>
    public List<string> ConnectionIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the node creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the node tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the node category
    /// </summary>
    public string Category { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the node description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the node importance (0.0 to 1.0)
    /// </summary>
    public double Importance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the node activation level (0.0 to 1.0)
    /// </summary>
    public double ActivationLevel { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the node last activation timestamp
    /// </summary>
    public DateTime? LastActivationTimestamp { get; set; }
    
    /// <summary>
    /// Adds a connection
    /// </summary>
    /// <param name="connectionId">The connection ID</param>
    public void AddConnection(string connectionId)
    {
        if (!ConnectionIds.Contains(connectionId))
        {
            ConnectionIds.Add(connectionId);
        }
    }
    
    /// <summary>
    /// Adds a tag
    /// </summary>
    /// <param name="tag">The tag</param>
    public void AddTag(string tag)
    {
        if (!Tags.Contains(tag))
        {
            Tags.Add(tag);
        }
    }
    
    /// <summary>
    /// Sets the attribute
    /// </summary>
    /// <param name="attribute">The attribute</param>
    /// <param name="value">The value</param>
    public void SetAttribute(string attribute, double value)
    {
        Attributes[attribute] = value;
    }
    
    /// <summary>
    /// Activates the node
    /// </summary>
    /// <param name="activationLevel">The activation level</param>
    public void Activate(double activationLevel)
    {
        ActivationLevel = activationLevel;
        LastActivationTimestamp = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Decays the activation level
    /// </summary>
    /// <param name="decayRate">The decay rate</param>
    public void DecayActivation(double decayRate)
    {
        ActivationLevel = Math.Max(0.0, ActivationLevel - decayRate);
    }
}
