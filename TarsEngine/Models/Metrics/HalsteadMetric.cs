using System;

namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a Halstead complexity metric
/// </summary>
public class HalsteadMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="HalsteadMetric"/> class
    /// </summary>
    public HalsteadMetric()
    {
        Category = MetricCategory.Complexity;
    }

    /// <summary>
    /// Gets or sets the Halstead type
    /// </summary>
    public HalsteadType Type { get; set; }

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
    /// Gets or sets the number of distinct operators
    /// </summary>
    public int DistinctOperators { get; set; }

    /// <summary>
    /// Gets or sets the number of distinct operands
    /// </summary>
    public int DistinctOperands { get; set; }

    /// <summary>
    /// Gets or sets the total number of operators
    /// </summary>
    public int TotalOperators { get; set; }

    /// <summary>
    /// Gets or sets the total number of operands
    /// </summary>
    public int TotalOperands { get; set; }

    /// <summary>
    /// Gets the program vocabulary (n1 + n2)
    /// </summary>
    public int Vocabulary => DistinctOperators + DistinctOperands;

    /// <summary>
    /// Gets the program length (N1 + N2)
    /// </summary>
    public int Length => TotalOperators + TotalOperands;

    /// <summary>
    /// Gets the calculated program volume
    /// </summary>
    public double Volume => Length * Math.Log2(Vocabulary);

    /// <summary>
    /// Gets the calculated program difficulty
    /// </summary>
    public double Difficulty => (DistinctOperators / 2.0) * (TotalOperands / (double)DistinctOperands);

    /// <summary>
    /// Gets the calculated program effort
    /// </summary>
    public double Effort => Difficulty * Volume;

    /// <summary>
    /// Gets the calculated time required to program (in seconds)
    /// </summary>
    public double TimeRequired => Effort / 18.0;

    /// <summary>
    /// Gets the calculated number of delivered bugs
    /// </summary>
    public double DeliveredBugs => Math.Pow(Effort, 2.0/3.0) / 3000.0;

    /// <summary>
    /// Gets or sets the baseline value for this Halstead metric
    /// </summary>
    public double BaselineValue { get; set; }

    /// <summary>
    /// Gets or sets the threshold value for this Halstead metric
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
    /// Gets whether the value is above the threshold
    /// </summary>
    public bool IsAboveThreshold => Value > ThresholdValue && ThresholdValue > 0;

    public string Description { get; set; } = string.Empty;
}

/// <summary>
/// Represents a Halstead metric type
/// </summary>
public enum HalsteadType
{
    /// <summary>
    /// Program vocabulary (n)
    /// </summary>
    Vocabulary,

    /// <summary>
    /// Program length (N)
    /// </summary>
    Length,

    /// <summary>
    /// Program volume (V)
    /// </summary>
    Volume,

    /// <summary>
    /// Program difficulty (D)
    /// </summary>
    Difficulty,

    /// <summary>
    /// Program effort (E)
    /// </summary>
    Effort,

    /// <summary>
    /// Time required to program (T)
    /// </summary>
    TimeRequired,

    /// <summary>
    /// Number of delivered bugs (B)
    /// </summary>
    DeliveredBugs
}
