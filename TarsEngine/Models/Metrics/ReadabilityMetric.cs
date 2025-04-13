namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a readability metric
/// </summary>
public class ReadabilityMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ReadabilityMetric"/> class
    /// </summary>
    public ReadabilityMetric()
    {
        Category = MetricCategory.Readability;
    }

    /// <summary>
    /// Gets or sets the readability type
    /// </summary>
    public ReadabilityType Type { get; set; }

    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;

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
    /// Gets or sets the lines of code
    /// </summary>
    public int LinesOfCode { get; set; }

    /// <summary>
    /// Gets or sets the comment percentage
    /// </summary>
    public double CommentPercentage { get; set; }

    /// <summary>
    /// Gets or sets the average identifier length
    /// </summary>
    public double AverageIdentifierLength { get; set; }

    /// <summary>
    /// Gets or sets the average line length
    /// </summary>
    public double AverageLineLength { get; set; }

    /// <summary>
    /// Gets or sets the maximum nesting depth
    /// </summary>
    public int MaxNestingDepth { get; set; }

    /// <summary>
    /// Gets or sets the average nesting depth
    /// </summary>
    public double AverageNestingDepth { get; set; }

    /// <summary>
    /// Gets or sets the number of long methods (methods with more than 30 lines)
    /// </summary>
    public int LongMethodCount { get; set; }

    /// <summary>
    /// Gets or sets the number of long lines (lines with more than 100 characters)
    /// </summary>
    public int LongLineCount { get; set; }

    /// <summary>
    /// Gets or sets the number of complex expressions (expressions with more than 3 operators)
    /// </summary>
    public int ComplexExpressionCount { get; set; }

    /// <summary>
    /// Gets or sets the number of magic numbers (numeric literals that are not 0, 1, or -1)
    /// </summary>
    public int MagicNumberCount { get; set; }

    /// <summary>
    /// Gets or sets the number of poorly named identifiers (identifiers that are too short or unclear)
    /// </summary>
    public int PoorlyNamedIdentifierCount { get; set; }

    /// <summary>
    /// Gets or sets the threshold value for this readability metric
    /// </summary>
    public double ThresholdValue { get; set; }

    /// <summary>
    /// Gets the readability score based on the readability type
    /// </summary>
    public double ReadabilityScore
    {
        get
        {
            return Type switch
            {
                ReadabilityType.IdentifierQuality => CalculateIdentifierQualityScore(),
                ReadabilityType.CommentQuality => CalculateCommentQualityScore(),
                ReadabilityType.CodeStructure => CalculateCodeStructureScore(),
                ReadabilityType.Overall => CalculateOverallReadabilityScore(),
                _ => 0
            };
        }
    }

    /// <summary>
    /// Gets the readability level based on the readability score
    /// </summary>
    public ReadabilityLevel ReadabilityLevel
    {
        get
        {
            if (ReadabilityScore >= 80)
            {
                return ReadabilityLevel.Excellent;
            }
            else if (ReadabilityScore >= 60)
            {
                return ReadabilityLevel.Good;
            }
            else if (ReadabilityScore >= 40)
            {
                return ReadabilityLevel.Fair;
            }
            else if (ReadabilityScore >= 20)
            {
                return ReadabilityLevel.Poor;
            }
            else
            {
                return ReadabilityLevel.VeryPoor;
            }
        }
    }

    /// <summary>
    /// Gets whether the value is below the threshold
    /// </summary>
    public bool IsBelowThreshold => Value < ThresholdValue && ThresholdValue > 0;

    /// <summary>
    /// Calculates the identifier quality score
    /// </summary>
    /// <returns>Identifier quality score</returns>
    private double CalculateIdentifierQualityScore()
    {
        // Start with a perfect score
        double score = 100;

        // Penalize for poorly named identifiers
        if (PoorlyNamedIdentifierCount > 0)
        {
            // Each poorly named identifier reduces the score by 5 points, up to 50 points
            score -= Math.Min(PoorlyNamedIdentifierCount * 5, 50);
        }

        // Penalize for very short average identifier length (less than 3 characters)
        if (AverageIdentifierLength < 3)
        {
            score -= (3 - AverageIdentifierLength) * 10;
        }

        // Penalize for very long average identifier length (more than 30 characters)
        if (AverageIdentifierLength > 30)
        {
            score -= (AverageIdentifierLength - 30) * 2;
        }

        // Ensure score is between 0 and 100
        return Math.Max(0, Math.Min(100, score));
    }

    /// <summary>
    /// Calculates the comment quality score
    /// </summary>
    /// <returns>Comment quality score</returns>
    private double CalculateCommentQualityScore()
    {
        // Start with a score based on comment percentage
        double score = 0;

        // Ideal comment percentage is between 15% and 30%
        if (CommentPercentage >= 15 && CommentPercentage <= 30)
        {
            score = 100;
        }
        else if (CommentPercentage < 15)
        {
            // Less than 15% comments reduces the score proportionally
            score = (CommentPercentage / 15) * 100;
        }
        else if (CommentPercentage > 30)
        {
            // More than 30% comments reduces the score proportionally, but less severely
            score = 100 - ((CommentPercentage - 30) / 70) * 50;
        }

        // Ensure score is between 0 and 100
        return Math.Max(0, Math.Min(100, score));
    }

    /// <summary>
    /// Calculates the code structure score
    /// </summary>
    /// <returns>Code structure score</returns>
    private double CalculateCodeStructureScore()
    {
        // Start with a perfect score
        double score = 100;

        // Penalize for deep nesting
        if (MaxNestingDepth > 3)
        {
            // Each level of nesting beyond 3 reduces the score by 10 points
            score -= (MaxNestingDepth - 3) * 10;
        }

        // Penalize for long methods
        if (LongMethodCount > 0)
        {
            // Each long method reduces the score by 5 points, up to 30 points
            score -= Math.Min(LongMethodCount * 5, 30);
        }

        // Penalize for long lines
        if (LongLineCount > 0)
        {
            // Each long line reduces the score by 2 points, up to 20 points
            score -= Math.Min(LongLineCount * 2, 20);
        }

        // Penalize for complex expressions
        if (ComplexExpressionCount > 0)
        {
            // Each complex expression reduces the score by 3 points, up to 30 points
            score -= Math.Min(ComplexExpressionCount * 3, 30);
        }

        // Penalize for magic numbers
        if (MagicNumberCount > 0)
        {
            // Each magic number reduces the score by 2 points, up to 20 points
            score -= Math.Min(MagicNumberCount * 2, 20);
        }

        // Ensure score is between 0 and 100
        return Math.Max(0, Math.Min(100, score));
    }

    /// <summary>
    /// Calculates the overall readability score
    /// </summary>
    /// <returns>Overall readability score</returns>
    private double CalculateOverallReadabilityScore()
    {
        // Calculate individual scores
        double identifierScore = CalculateIdentifierQualityScore();
        double commentScore = CalculateCommentQualityScore();
        double structureScore = CalculateCodeStructureScore();

        // Overall score is a weighted average of the individual scores
        // Structure is weighted more heavily as it has the most impact on readability
        return (identifierScore * 0.3) + (commentScore * 0.3) + (structureScore * 0.4);
    }
}

/// <summary>
/// Represents a readability type
/// </summary>
public enum ReadabilityType
{
    /// <summary>
    /// Identifier quality (naming conventions, length, clarity)
    /// </summary>
    IdentifierQuality,

    /// <summary>
    /// Comment quality (coverage, clarity, usefulness)
    /// </summary>
    CommentQuality,

    /// <summary>
    /// Code structure (nesting, method length, line length, complexity)
    /// </summary>
    CodeStructure,

    /// <summary>
    /// Overall readability (combination of all factors)
    /// </summary>
    Overall
}

/// <summary>
/// Represents a readability level
/// </summary>
public enum ReadabilityLevel
{
    /// <summary>
    /// Excellent readability (score greater than or equal to 80)
    /// </summary>
    Excellent,

    /// <summary>
    /// Good readability (60 less than or equal to score less than 80)
    /// </summary>
    Good,

    /// <summary>
    /// Fair readability (40 less than or equal to score less than 60)
    /// </summary>
    Fair,

    /// <summary>
    /// Poor readability (20 less than or equal to score less than 40)
    /// </summary>
    Poor,

    /// <summary>
    /// Very poor readability (score less than 20)
    /// </summary>
    VeryPoor
}
