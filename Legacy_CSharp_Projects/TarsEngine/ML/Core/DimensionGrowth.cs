namespace TarsEngine.ML.Core;

/// <summary>
/// Represents the growth of an intelligence dimension
/// </summary>
public class DimensionGrowth
{
    /// <summary>
    /// Gets or sets the dimension name
    /// </summary>
    public string Dimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the growth rate (units per day)
    /// </summary>
    public double GrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the current value
    /// </summary>
    public double CurrentValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic growth rate
    /// </summary>
    public double LogGrowthRate { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic current value
    /// </summary>
    public double LogCurrentValue { get; set; }
    
    /// <summary>
    /// Gets the doubling time in days (how long it takes to double the current value)
    /// </summary>
    public double DoublingTimeDays => GrowthRate > 0 ? CurrentValue / GrowthRate : double.PositiveInfinity;
    
    /// <summary>
    /// Gets the logarithmic doubling time in days
    /// </summary>
    public double LogDoublingTimeDays => LogGrowthRate > 0 ? Math.Log10(2) / LogGrowthRate : double.PositiveInfinity;
    
    /// <summary>
    /// Gets a description of the dimension growth
    /// </summary>
    /// <returns>A description of the dimension growth</returns>
    public string GetDescription()
    {
        if (GrowthRate > 0)
        {
            return $"{Dimension}: +{GrowthRate:F3} per day (doubles in {DoublingTimeDays:F1} days)";
        }
        else if (GrowthRate < 0)
        {
            return $"{Dimension}: {GrowthRate:F3} per day (declining)";
        }
        else
        {
            return $"{Dimension}: No growth";
        }
    }
    
    /// <summary>
    /// Gets a logarithmic description of the dimension growth
    /// </summary>
    /// <returns>A logarithmic description of the dimension growth</returns>
    public string GetLogarithmicDescription()
    {
        if (LogGrowthRate > 0)
        {
            return $"{Dimension}: +{LogGrowthRate:F3} log units per day (order of magnitude increase in {Math.Log10(10) / LogGrowthRate:F1} days)";
        }
        else if (LogGrowthRate < 0)
        {
            return $"{Dimension}: {LogGrowthRate:F3} log units per day (declining)";
        }
        else
        {
            return $"{Dimension}: No logarithmic growth";
        }
    }
}
