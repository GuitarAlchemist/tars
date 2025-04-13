namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a relationship between elements in a problem representation
/// </summary>
public class ElementRelationship
{
    /// <summary>
    /// Gets or sets the relationship ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the source element ID
    /// </summary>
    public string SourceId { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target element ID
    /// </summary>
    public string TargetId { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the relationship type
    /// </summary>
    public RelationshipType Type { get; set; } = RelationshipType.Association;
    
    /// <summary>
    /// Gets or sets the relationship strength (0.0 to 1.0)
    /// </summary>
    public double Strength { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the relationship properties
    /// </summary>
    public Dictionary<string, object> Properties { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the relationship bidirectional flag
    /// </summary>
    public bool IsBidirectional { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the relationship description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Adds a property
    /// </summary>
    /// <param name="name">The property name</param>
    /// <param name="value">The property value</param>
    public void AddProperty(string name, object value)
    {
        Properties[name] = value;
    }
    
    /// <summary>
    /// Creates a clone of the relationship
    /// </summary>
    /// <returns>The cloned relationship</returns>
    public ElementRelationship Clone()
    {
        return new ElementRelationship
        {
            SourceId = SourceId,
            TargetId = TargetId,
            Type = Type,
            Strength = Strength,
            Properties = new Dictionary<string, object>(Properties),
            IsBidirectional = IsBidirectional,
            Description = Description
        };
    }
}
