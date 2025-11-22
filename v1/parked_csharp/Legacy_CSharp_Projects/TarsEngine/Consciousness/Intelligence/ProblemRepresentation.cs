namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a problem representation for restructuring
/// </summary>
public class ProblemRepresentation
{
    /// <summary>
    /// Gets or sets the representation ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the problem statement
    /// </summary>
    public string ProblemStatement { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the representation type
    /// </summary>
    public RepresentationType Type { get; set; } = RepresentationType.Standard;
    
    /// <summary>
    /// Gets or sets the representation elements
    /// </summary>
    public List<RepresentationElement> Elements { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the representation constraints
    /// </summary>
    public List<string> Constraints { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the representation assumptions
    /// </summary>
    public List<string> Assumptions { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the representation creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the representation context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the representation tags
    /// </summary>
    public List<string> Tags { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the representation effectiveness (0.0 to 1.0)
    /// </summary>
    public double Effectiveness { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the representation novelty (0.0 to 1.0)
    /// </summary>
    public double Novelty { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the representation parent ID
    /// </summary>
    public string? ParentId { get; set; }
    
    /// <summary>
    /// Gets or sets the representation child IDs
    /// </summary>
    public List<string> ChildIds { get; set; } = [];
    
    /// <summary>
    /// Adds an element
    /// </summary>
    /// <param name="element">The element</param>
    public void AddElement(RepresentationElement element)
    {
        Elements.Add(element);
    }
    
    /// <summary>
    /// Adds a constraint
    /// </summary>
    /// <param name="constraint">The constraint</param>
    public void AddConstraint(string constraint)
    {
        Constraints.Add(constraint);
    }
    
    /// <summary>
    /// Adds an assumption
    /// </summary>
    /// <param name="assumption">The assumption</param>
    public void AddAssumption(string assumption)
    {
        Assumptions.Add(assumption);
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
    /// Adds a child
    /// </summary>
    /// <param name="childId">The child ID</param>
    public void AddChild(string childId)
    {
        if (!ChildIds.Contains(childId))
        {
            ChildIds.Add(childId);
        }
    }
    
    /// <summary>
    /// Creates a clone of the representation
    /// </summary>
    /// <returns>The cloned representation</returns>
    public ProblemRepresentation Clone()
    {
        var clone = new ProblemRepresentation
        {
            ProblemStatement = ProblemStatement,
            Type = Type,
            Constraints = [..Constraints],
            Assumptions = [..Assumptions],
            Context = new Dictionary<string, object>(Context),
            Tags = [..Tags],
            Effectiveness = Effectiveness,
            Novelty = Novelty,
            ParentId = Id
        };
        
        // Clone elements
        foreach (var element in Elements)
        {
            clone.Elements.Add(element.Clone());
        }
        
        return clone;
    }
}
