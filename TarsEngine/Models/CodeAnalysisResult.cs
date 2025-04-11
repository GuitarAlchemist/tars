using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents the result of a code analysis
/// </summary>
public class CodeAnalysisResult
{
    /// <summary>
    /// Gets or sets the ID of the analysis
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the path to the analyzed file or directory
    /// </summary>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the programming language of the analyzed code
    /// </summary>
    public string Language { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the analysis was performed
    /// </summary>
    public DateTime AnalyzedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the list of code issues found during analysis
    /// </summary>
    public List<CodeIssue> Issues { get; set; } = new List<CodeIssue>();

    /// <summary>
    /// Gets or sets the list of code metrics calculated during analysis
    /// </summary>
    public List<CodeMetric> Metrics { get; set; } = new List<CodeMetric>();

    /// <summary>
    /// Gets or sets the list of code structures identified during analysis
    /// </summary>
    public List<CodeStructure> Structures { get; set; } = new List<CodeStructure>();

    /// <summary>
    /// Gets or sets whether the analysis was successful
    /// </summary>
    public bool IsSuccessful { get; set; } = true;

    /// <summary>
    /// Gets or sets the list of errors that occurred during analysis
    /// </summary>
    public List<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets additional metadata about the analysis
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}

/// <summary>
/// Represents an issue found in code during analysis
/// </summary>
public class CodeIssue
{
    /// <summary>
    /// Gets or sets the ID of the issue
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the type of the issue
    /// </summary>
    public CodeIssueType Type { get; set; }

    /// <summary>
    /// Gets or sets the severity of the issue
    /// </summary>
    public IssueSeverity Severity { get; set; }

    /// <summary>
    /// Gets or sets the title of the issue
    /// </summary>
    public string Title { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the issue
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location of the issue in the code
    /// </summary>
    public CodeLocation Location { get; set; } = new CodeLocation();

    /// <summary>
    /// Gets or sets the code snippet containing the issue
    /// </summary>
    public string CodeSnippet { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the suggested fix for the issue
    /// </summary>
    public string SuggestedFix { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the impact score of the issue (0.0 to 1.0)
    /// </summary>
    public double ImpactScore { get; set; }

    /// <summary>
    /// Gets or sets the fix difficulty score of the issue (0.0 to 1.0)
    /// </summary>
    public double FixDifficultyScore { get; set; }

    /// <summary>
    /// Gets or sets additional tags for the issue
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
}

/// <summary>
/// Represents a metric calculated for code during analysis
/// </summary>
public class CodeMetric
{
    /// <summary>
    /// Gets or sets the name of the metric
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the value of the metric
    /// </summary>
    public double Value { get; set; }

    /// <summary>
    /// Gets or sets the type of the metric
    /// </summary>
    public MetricType Type { get; set; }

    /// <summary>
    /// Gets or sets the scope of the metric
    /// </summary>
    public MetricScope Scope { get; set; }

    /// <summary>
    /// Gets or sets the target name (class, method, etc.) for the metric
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location of the code for which the metric was calculated
    /// </summary>
    public CodeLocation Location { get; set; } = new CodeLocation();

    /// <summary>
    /// Gets or sets the threshold for good values of this metric
    /// </summary>
    public double GoodThreshold { get; set; }

    /// <summary>
    /// Gets or sets the threshold for acceptable values of this metric
    /// </summary>
    public double AcceptableThreshold { get; set; }

    /// <summary>
    /// Gets or sets the threshold for poor values of this metric
    /// </summary>
    public double PoorThreshold { get; set; }

    /// <summary>
    /// Gets the quality rating of the metric value
    /// </summary>
    public MetricQuality Quality
    {
        get
        {
            if (Value <= GoodThreshold)
            {
                return MetricQuality.Good;
            }
            else if (Value <= AcceptableThreshold)
            {
                return MetricQuality.Acceptable;
            }
            else
            {
                return MetricQuality.Poor;
            }
        }
    }
}

/// <summary>
/// Represents a code structure identified during analysis
/// </summary>
public class CodeStructure
{
    /// <summary>
    /// Gets or sets the type of the structure
    /// </summary>
    public StructureType Type { get; set; }

    /// <summary>
    /// Gets or sets the name of the structure
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location of the structure in the code
    /// </summary>
    public CodeLocation Location { get; set; } = new CodeLocation();

    /// <summary>
    /// Gets or sets the parent structure, if any
    /// </summary>
    public string? ParentName { get; set; }

    /// <summary>
    /// Gets or sets the list of child structure names
    /// </summary>
    public List<string> ChildNames { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of dependencies for this structure
    /// </summary>
    public List<string> Dependencies { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the complexity score of the structure
    /// </summary>
    public double ComplexityScore { get; set; }

    /// <summary>
    /// Gets or sets the size of the structure (lines of code)
    /// </summary>
    public int Size { get; set; }

    /// <summary>
    /// Gets or sets additional properties of the structure
    /// </summary>
    public Dictionary<string, string> Properties { get; set; } = new Dictionary<string, string>();
}

/// <summary>
/// Represents a location in code
/// </summary>
public class CodeLocation
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the starting line number
    /// </summary>
    public int StartLine { get; set; }

    /// <summary>
    /// Gets or sets the ending line number
    /// </summary>
    public int EndLine { get; set; }

    /// <summary>
    /// Gets or sets the starting column number
    /// </summary>
    public int StartColumn { get; set; }

    /// <summary>
    /// Gets or sets the ending column number
    /// </summary>
    public int EndColumn { get; set; }

    /// <summary>
    /// Gets or sets the namespace
    /// </summary>
    public string Namespace { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the class name
    /// </summary>
    public string ClassName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the method name
    /// </summary>
    public string MethodName { get; set; } = string.Empty;
}

/// <summary>
/// Represents the type of a code issue
/// </summary>
public enum CodeIssueType
{
    /// <summary>
    /// Code smell
    /// </summary>
    CodeSmell,

    /// <summary>
    /// Bug
    /// </summary>
    Bug,

    /// <summary>
    /// Vulnerability
    /// </summary>
    Vulnerability,

    /// <summary>
    /// Security hotspot
    /// </summary>
    SecurityHotspot,

    /// <summary>
    /// Performance issue
    /// </summary>
    Performance,

    /// <summary>
    /// Maintainability issue
    /// </summary>
    Maintainability,

    /// <summary>
    /// Design issue
    /// </summary>
    Design,

    /// <summary>
    /// Documentation issue
    /// </summary>
    Documentation,

    /// <summary>
    /// Duplication
    /// </summary>
    Duplication,

    /// <summary>
    /// Complexity
    /// </summary>
    Complexity,

    /// <summary>
    /// Style
    /// </summary>
    Style,

    /// <summary>
    /// Naming
    /// </summary>
    Naming,

    /// <summary>
    /// Unused code
    /// </summary>
    UnusedCode,

    /// <summary>
    /// Dead code
    /// </summary>
    DeadCode,

    /// <summary>
    /// Security issue
    /// </summary>
    Security,

    /// <summary>
    /// Other issue
    /// </summary>
    Other
}

/// <summary>
/// Represents the severity of an issue
/// </summary>
public enum IssueSeverity
{
    /// <summary>
    /// Blocker severity
    /// </summary>
    Blocker,

    /// <summary>
    /// Critical severity
    /// </summary>
    Critical,

    /// <summary>
    /// Major severity
    /// </summary>
    Major,

    /// <summary>
    /// Minor severity
    /// </summary>
    Minor,

    /// <summary>
    /// Trivial severity
    /// </summary>
    Trivial,

    /// <summary>
    /// Error severity
    /// </summary>
    Error,

    /// <summary>
    /// Info severity
    /// </summary>
    Info
}

/// <summary>
/// Represents the type of a metric
/// </summary>
public enum MetricType
{
    /// <summary>
    /// Complexity metric
    /// </summary>
    Complexity,

    /// <summary>
    /// Size metric
    /// </summary>
    Size,

    /// <summary>
    /// Coupling metric
    /// </summary>
    Coupling,

    /// <summary>
    /// Cohesion metric
    /// </summary>
    Cohesion,

    /// <summary>
    /// Inheritance metric
    /// </summary>
    Inheritance,

    /// <summary>
    /// Maintainability metric
    /// </summary>
    Maintainability,

    /// <summary>
    /// Documentation metric
    /// </summary>
    Documentation,

    /// <summary>
    /// Test coverage metric
    /// </summary>
    TestCoverage,

    /// <summary>
    /// Performance metric
    /// </summary>
    Performance,

    /// <summary>
    /// Other metric
    /// </summary>
    Other
}

/// <summary>
/// Represents the scope of a metric
/// </summary>
public enum MetricScope
{
    /// <summary>
    /// Method scope
    /// </summary>
    Method,

    /// <summary>
    /// Class scope
    /// </summary>
    Class,

    /// <summary>
    /// Namespace scope
    /// </summary>
    Namespace,

    /// <summary>
    /// File scope
    /// </summary>
    File,

    /// <summary>
    /// Project scope
    /// </summary>
    Project,

    /// <summary>
    /// Solution scope
    /// </summary>
    Solution
}

/// <summary>
/// Represents the quality of a metric value
/// </summary>
public enum MetricQuality
{
    /// <summary>
    /// Good quality
    /// </summary>
    Good,

    /// <summary>
    /// Acceptable quality
    /// </summary>
    Acceptable,

    /// <summary>
    /// Poor quality
    /// </summary>
    Poor
}

/// <summary>
/// Represents the type of a code structure
/// </summary>
public enum StructureType
{
    /// <summary>
    /// Namespace
    /// </summary>
    Namespace,

    /// <summary>
    /// Class
    /// </summary>
    Class,

    /// <summary>
    /// Interface
    /// </summary>
    Interface,

    /// <summary>
    /// Struct
    /// </summary>
    Struct,

    /// <summary>
    /// Enum
    /// </summary>
    Enum,

    /// <summary>
    /// Method
    /// </summary>
    Method,

    /// <summary>
    /// Property
    /// </summary>
    Property,

    /// <summary>
    /// Field
    /// </summary>
    Field,

    /// <summary>
    /// Event
    /// </summary>
    Event,

    /// <summary>
    /// Delegate
    /// </summary>
    Delegate,

    /// <summary>
    /// Record
    /// </summary>
    Record,

    /// <summary>
    /// Module (F#)
    /// </summary>
    Module,

    /// <summary>
    /// Function (F#)
    /// </summary>
    Function,

    /// <summary>
    /// Type (F#)
    /// </summary>
    Type,

    /// <summary>
    /// Union (F#)
    /// </summary>
    Union,

    /// <summary>
    /// Other structure
    /// </summary>
    Other
}
