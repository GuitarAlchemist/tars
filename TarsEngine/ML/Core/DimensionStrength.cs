using System;

namespace TarsEngine.ML.Core;

/// <summary>
/// Represents the strength of an intelligence dimension
/// </summary>
public class DimensionStrength
{
    /// <summary>
    /// Gets or sets the dimension name
    /// </summary>
    public string Dimension { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the current value
    /// </summary>
    public double Value { get; set; }
    
    /// <summary>
    /// Gets or sets the relative strength compared to human baseline
    /// </summary>
    public double RelativeStrength { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic value
    /// </summary>
    public double LogValue { get; set; }
    
    /// <summary>
    /// Gets or sets the logarithmic relative strength
    /// </summary>
    public double LogRelativeStrength { get; set; }
    
    /// <summary>
    /// Gets a description of the dimension strength
    /// </summary>
    /// <returns>A description of the dimension strength</returns>
    public string GetDescription()
    {
        if (RelativeStrength > 1.0)
        {
            return $"{Dimension}: {Value:F2} ({RelativeStrength:F2}x human baseline)";
        }
        else
        {
            return $"{Dimension}: {Value:F2} ({RelativeStrength * 100:F1}% of human baseline)";
        }
    }
    
    /// <summary>
    /// Gets a logarithmic description of the dimension strength
    /// </summary>
    /// <returns>A logarithmic description of the dimension strength</returns>
    public string GetLogarithmicDescription()
    {
        if (LogRelativeStrength > 0)
        {
            return $"{Dimension}: {LogValue:F2} log units ({Math.Pow(10, LogRelativeStrength):F2}x human baseline)";
        }
        else
        {
            return $"{Dimension}: {LogValue:F2} log units ({Math.Pow(10, LogRelativeStrength) * 100:F1}% of human baseline)";
        }
    }
}
