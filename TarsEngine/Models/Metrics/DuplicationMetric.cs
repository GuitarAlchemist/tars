using System;
using System.Collections.Generic;

namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a code duplication metric
/// </summary>
public class DuplicationMetric : BaseMetric
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DuplicationMetric"/> class
    /// </summary>
    public DuplicationMetric()
    {
        Category = MetricCategory.Duplication;
        DuplicatedBlocks = new List<DuplicatedBlock>();
    }
    
    /// <summary>
    /// Gets or sets the duplication type
    /// </summary>
    public DuplicationType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the language
    /// </summary>
    public string Language { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target (method, class, file, etc.)
    /// </summary>
    public string Target { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target type (method, class, file, etc.)
    /// </summary>
    public TargetType TargetType { get; set; }
    
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the total lines of code
    /// </summary>
    public int TotalLinesOfCode { get; set; }
    
    /// <summary>
    /// Gets or sets the duplicated lines of code
    /// </summary>
    public int DuplicatedLinesOfCode { get; set; }
    
    /// <summary>
    /// Gets or sets the duplication percentage
    /// </summary>
    public double DuplicationPercentage { get; set; }
    
    /// <summary>
    /// Gets or sets the number of duplicated blocks
    /// </summary>
    public int DuplicatedBlockCount { get; set; }
    
    /// <summary>
    /// Gets or sets the list of duplicated blocks
    /// </summary>
    public List<DuplicatedBlock> DuplicatedBlocks { get; set; }
    
    /// <summary>
    /// Gets or sets the threshold value for this duplication metric
    /// </summary>
    public double ThresholdValue { get; set; }
    
    /// <summary>
    /// Gets whether the value is above the threshold
    /// </summary>
    public bool IsAboveThreshold => Value > ThresholdValue && ThresholdValue > 0;
    
    /// <summary>
    /// Gets the duplication level based on the duplication percentage
    /// </summary>
    public DuplicationLevel DuplicationLevel
    {
        get
        {
            if (DuplicationPercentage < 3)
            {
                return DuplicationLevel.Low;
            }
            else if (DuplicationPercentage < 10)
            {
                return DuplicationLevel.Moderate;
            }
            else if (DuplicationPercentage < 20)
            {
                return DuplicationLevel.High;
            }
            else
            {
                return DuplicationLevel.VeryHigh;
            }
        }
    }
}

/// <summary>
/// Represents a duplicated block of code
/// </summary>
public class DuplicatedBlock
{
    /// <summary>
    /// Gets or sets the source file path
    /// </summary>
    public string SourceFilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the source start line
    /// </summary>
    public int SourceStartLine { get; set; }
    
    /// <summary>
    /// Gets or sets the source end line
    /// </summary>
    public int SourceEndLine { get; set; }
    
    /// <summary>
    /// Gets or sets the target file path
    /// </summary>
    public string TargetFilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target start line
    /// </summary>
    public int TargetStartLine { get; set; }
    
    /// <summary>
    /// Gets or sets the target end line
    /// </summary>
    public int TargetEndLine { get; set; }
    
    /// <summary>
    /// Gets or sets the duplicated code
    /// </summary>
    public string DuplicatedCode { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the similarity percentage
    /// </summary>
    public double SimilarityPercentage { get; set; }
    
    /// <summary>
    /// Gets the number of duplicated lines
    /// </summary>
    public int DuplicatedLines => SourceEndLine - SourceStartLine + 1;
}

/// <summary>
/// Represents a duplication type
/// </summary>
public enum DuplicationType
{
    /// <summary>
    /// Token-based duplication (exact matches)
    /// </summary>
    TokenBased,
    
    /// <summary>
    /// Semantic duplication (similar functionality)
    /// </summary>
    Semantic,
    
    /// <summary>
    /// Overall duplication (combination of all types)
    /// </summary>
    Overall
}

/// <summary>
/// Represents a duplication level
/// </summary>
public enum DuplicationLevel
{
    /// <summary>
    /// Low duplication (less than 3%)
    /// </summary>
    Low,
    
    /// <summary>
    /// Moderate duplication (3-10%)
    /// </summary>
    Moderate,
    
    /// <summary>
    /// High duplication (10-20%)
    /// </summary>
    High,
    
    /// <summary>
    /// Very high duplication (more than 20%)
    /// </summary>
    VeryHigh
}
