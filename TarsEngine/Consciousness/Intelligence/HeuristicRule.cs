namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a heuristic rule
/// </summary>
public class HeuristicRule
{
    /// <summary>
    /// Gets or sets the rule ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the rule name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the rule description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the rule reliability (0.0 to 1.0)
    /// </summary>
    public double Reliability { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the rule context
    /// </summary>
    public string Context { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the rule creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the rule last used timestamp
    /// </summary>
    public DateTime? LastUsedTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the rule usage count
    /// </summary>
    public int UsageCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the rule success count
    /// </summary>
    public int SuccessCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the rule tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the rule examples
    /// </summary>
    public List<string> Examples { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the rule counter-examples
    /// </summary>
    public List<string> CounterExamples { get; set; } = new();
    
    /// <summary>
    /// Gets the rule success rate
    /// </summary>
    public double SuccessRate => UsageCount > 0 ? (double)SuccessCount / UsageCount : 0.0;
    
    /// <summary>
    /// Records a rule usage
    /// </summary>
    /// <param name="success">Whether the usage was successful</param>
    public void RecordUsage(bool success)
    {
        UsageCount++;
        if (success)
        {
            SuccessCount++;
        }
        LastUsedTimestamp = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Adds an example
    /// </summary>
    /// <param name="example">The example</param>
    /// <param name="isCounterExample">Whether the example is a counter-example</param>
    public void AddExample(string example, bool isCounterExample = false)
    {
        if (isCounterExample)
        {
            CounterExamples.Add(example);
        }
        else
        {
            Examples.Add(example);
        }
    }
    
    /// <summary>
    /// Updates the rule reliability
    /// </summary>
    /// <param name="reliability">The reliability</param>
    public void UpdateReliability(double reliability)
    {
        Reliability = Math.Max(0.0, Math.Min(1.0, reliability));
    }
}
