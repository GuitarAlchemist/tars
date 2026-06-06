namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a maintainability index metric
/// </summary>
public class MaintainabilityMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MaintainabilityMetric"/> class
    /// </summary>
    public MaintainabilityMetric()
    {
        Category = MetricCategory.Complexity;
    }

    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the language
    /// </summary>
    public string Language { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the target (method, class, etc.)
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the target type (method, class, etc.)
    /// </summary>
    public TargetType TargetType { get; set; }

    /// <summary>
    /// Gets or sets the Halstead volume
    /// </summary>
    public double HalsteadVolume { get; set; }

    /// <summary>
    /// Gets or sets the cyclomatic complexity
    /// </summary>
    public double CyclomaticComplexity { get; set; }

    /// <summary>
    /// Gets or sets the lines of code
    /// </summary>
    public int LinesOfCode { get; set; }

    /// <summary>
    /// Gets or sets the comment percentage
    /// </summary>
    public double CommentPercentage { get; set; }

    /// <summary>
    /// Gets or sets whether to use the original maintainability index formula
    /// </summary>
    public bool UseOriginalFormula { get; set; }

    /// <summary>
    /// Gets or sets whether to use the Microsoft Visual Studio maintainability index formula
    /// </summary>
    public bool UseMicrosoftFormula { get; set; }

    /// <summary>
    /// Gets or sets whether to include comments in the calculation
    /// </summary>
    public bool IncludeComments { get; set; } = true;

    /// <summary>
    /// Gets the calculated maintainability index using the original formula
    /// </summary>
    public double OriginalMaintainabilityIndex
    {
        get
        {
            // Original formula: MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CM))
            // Where:
            // V = Halstead Volume
            // G = Cyclomatic Complexity
            // LOC = Lines of Code
            // CM = Comment Percentage (0-1)

            double mi = 171;

            if (HalsteadVolume > 0)
            {
                mi -= 5.2 * Math.Log(HalsteadVolume);
            }

            mi -= 0.23 * CyclomaticComplexity;

            if (LinesOfCode > 0)
            {
                mi -= 16.2 * Math.Log(LinesOfCode);
            }

            if (IncludeComments && CommentPercentage > 0)
            {
                mi += 50 * Math.Sin(Math.Sqrt(2.4 * CommentPercentage / 100.0));
            }

            return Math.Max(0, Math.Min(100, mi));
        }
    }

    /// <summary>
    /// Gets the calculated maintainability index using the Microsoft formula
    /// </summary>
    public double MicrosoftMaintainabilityIndex
    {
        get
        {
            // Microsoft formula: MI = MAX(0, (171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)) * 100 / 171)
            // Where:
            // V = Halstead Volume
            // G = Cyclomatic Complexity
            // LOC = Lines of Code

            double mi = 171;

            if (HalsteadVolume > 0)
            {
                mi -= 5.2 * Math.Log(HalsteadVolume);
            }

            mi -= 0.23 * CyclomaticComplexity;

            if (LinesOfCode > 0)
            {
                mi -= 16.2 * Math.Log(LinesOfCode);
            }

            return Math.Max(0, Math.Min(100, mi * 100 / 171));
        }
    }

    /// <summary>
    /// Gets the calculated maintainability index
    /// </summary>
    public double MaintainabilityIndex
    {
        get
        {
            if (UseOriginalFormula)
            {
                return OriginalMaintainabilityIndex;
            }
            else if (UseMicrosoftFormula)
            {
                return MicrosoftMaintainabilityIndex;
            }
            else
            {
                // Default to Microsoft formula
                return MicrosoftMaintainabilityIndex;
            }
        }
    }

    /// <summary>
    /// Gets the maintainability level based on the maintainability index
    /// </summary>
    public MaintainabilityLevel MaintainabilityLevel
    {
        get
        {
            if (MaintainabilityIndex >= 80)
            {
                return MaintainabilityLevel.High;
            }
            else if (MaintainabilityIndex >= 60)
            {
                return MaintainabilityLevel.Moderate;
            }
            else if (MaintainabilityIndex >= 40)
            {
                return MaintainabilityLevel.Low;
            }
            else
            {
                return MaintainabilityLevel.VeryLow;
            }
        }
    }

    /// <summary>
    /// Gets or sets the baseline value for this maintainability metric
    /// </summary>
    public double BaselineValue { get; set; }

    /// <summary>
    /// Gets or sets the threshold value for this maintainability metric
    /// </summary>
    public double ThresholdValue { get; set; }

    /// <summary>
    /// Gets the ratio compared to baseline
    /// </summary>
    public double BaselineRatio => BaselineValue > 0 ? Value / BaselineValue : 0;

    /// <summary>
    /// Gets the ratio compared to threshold
    /// </summary>
    public double ThresholdRatio => ThresholdValue > 0 ? Value / ThresholdValue : 0;

    /// <summary>
    /// Gets whether the value is below the threshold
    /// </summary>
    public bool IsBelowThreshold => Value < ThresholdValue && ThresholdValue > 0;

    public string Description { get; set; } = string.Empty;
}

/// <summary>
/// Represents a maintainability level
/// </summary>
public enum MaintainabilityLevel
{
    /// <summary>
    /// High maintainability (MI greater than or equal to 80)
    /// </summary>
    High,

    /// <summary>
    /// Moderate maintainability (60 less than or equal to MI less than 80)
    /// </summary>
    Moderate,

    /// <summary>
    /// Low maintainability (40 less than or equal to MI less than 60)
    /// </summary>
    Low,

    /// <summary>
    /// Very low maintainability (MI less than 40)
    /// </summary>
    VeryLow
}
