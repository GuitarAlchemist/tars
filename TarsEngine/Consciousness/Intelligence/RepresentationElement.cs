namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents an element in a problem representation
/// </summary>
public class RepresentationElement
{
    /// <summary>
    /// Gets or sets the element ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the element name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the element type
    /// </summary>
    public ElementType Type { get; set; } = ElementType.Entity;
    
    /// <summary>
    /// Gets or sets the element properties
    /// </summary>
    public Dictionary<string, object> Properties { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the element relationships
    /// </summary>
    public List<ElementRelationship> Relationships { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the element importance (0.0 to 1.0)
    /// </summary>
    public double Importance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the element flexibility (0.0 to 1.0)
    /// </summary>
    public double Flexibility { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the element context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
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
    /// Adds a relationship
    /// </summary>
    /// <param name="relationship">The relationship</param>
    public void AddRelationship(ElementRelationship relationship)
    {
        Relationships.Add(relationship);
    }
    
    /// <summary>
    /// Creates a relationship
    /// </summary>
    /// <param name="targetId">The target ID</param>
    /// <param name="type">The relationship type</param>
    /// <param name="strength">The relationship strength</param>
    /// <returns>The created relationship</returns>
    public ElementRelationship CreateRelationship(string targetId, RelationshipType type, double strength)
    {
        var relationship = new ElementRelationship
        {
            SourceId = Id,
            TargetId = targetId,
            Type = type,
            Strength = strength
        };
        
        Relationships.Add(relationship);
        
        return relationship;
    }
    
    /// <summary>
    /// Creates a clone of the element
    /// </summary>
    /// <returns>The cloned element</returns>
    public RepresentationElement Clone()
    {
        var clone = new RepresentationElement
        {
            Name = Name,
            Type = Type,
            Properties = new Dictionary<string, object>(Properties),
            Importance = Importance,
            Flexibility = Flexibility,
            Context = new Dictionary<string, object>(Context)
        };
        
        // Clone relationships
        foreach (var relationship in Relationships)
        {
            clone.Relationships.Add(relationship.Clone());
        }
        
        return clone;
    }
}
