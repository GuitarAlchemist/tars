namespace TarsEngine.ML.Core;

/// <summary>
/// Represents a comparison between TARS and human baseline for a specific intelligence dimension
/// </summary>
public class ComparisonWithHuman
{
    /// <summary>
    /// Gets or sets the intelligence dimension
    /// </summary>
    public string Dimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the current value
    /// </summary>
    public double CurrentValue { get; set; }
    
    /// <summary>
    /// Gets or sets the human baseline value
    /// </summary>
    public double HumanValue { get; set; }
    
    /// <summary>
    /// Gets or sets the ratio (current/human)
    /// </summary>
    public double Ratio { get; set; }
    
    /// <summary>
    /// Gets or sets the difference (current-human)
    /// </summary>
    public double Difference { get; set; }
    
    /// <summary>
    /// Gets or sets whether TARS has surpassed human in this dimension
    /// </summary>
    public bool HasSurpassedHuman { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic current value
    /// </summary>
    public double LogCurrentValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic human value
    /// </summary>
    public double LogHumanValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic ratio
    /// </summary>
    public double LogRatio { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic difference
    /// </summary>
    public double LogDifference { get; set; }
    
    /// <summary>
    /// Gets a description of the comparison
    /// </summary>
    /// <returns>A description of the comparison</returns>
    public string GetDescription()
    {
        if (HasSurpassedHuman)
        {
            return $"TARS has surpassed human baseline in {Dimension} by {Difference:F2} units (ratio: {Ratio:F2}x)";
        }
        else
        {
            return $"TARS is at {Ratio * 100:F1}% of human baseline in {Dimension} (difference: {Difference:F2} units)";
        }
    }
    
    /// <summary>
    /// Gets a logarithmic description of the comparison
    /// </summary>
    /// <returns>A logarithmic description of the comparison</returns>
    public string GetLogarithmicDescription()
    {
        if (HasSurpassedHuman)
        {
            return $"TARS has surpassed human baseline in {Dimension} by {LogDifference:F2} log units (log ratio: {LogRatio:F2})";
        }
        else
        {
            return $"TARS is at {Math.Pow(10, LogRatio) * 100:F1}% of human baseline in {Dimension} (log difference: {LogDifference:F2})";
        }
    }
}
