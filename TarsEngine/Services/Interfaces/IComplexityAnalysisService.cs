using UnifiedComplexityType = TarsEngine.Unified.ComplexityType;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the complexity analysis service
/// </summary>
public interface IComplexityAnalysisService
{
    /// <summary>
    /// Analyzes code complexity for a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="language">Programming language</param>
    /// <returns>Complexity analysis result</returns>
    Task<ComplexityAnalysisResult> AnalyzeComplexityAsync(string filePath, string language);

    /// <summary>
    /// Analyzes code complexity for a specific project
    /// </summary>
    /// <param name="projectPath">Path to the project to analyze</param>
    /// <returns>Complexity analysis result</returns>
    Task<ComplexityAnalysisResult> AnalyzeProjectComplexityAsync(string projectPath);

    /// <summary>
    /// Analyzes code complexity for a specific solution
    /// </summary>
    /// <param name="solutionPath">Path to the solution to analyze</param>
    /// <returns>Complexity analysis result</returns>
    Task<ComplexityAnalysisResult> AnalyzeSolutionComplexityAsync(string solutionPath);

    /// <summary>
    /// Identifies overly complex code
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="language">Programming language</param>
    /// <param name="threshold">Complexity threshold</param>
    /// <returns>List of complex code sections</returns>
    Task<List<ComplexCodeSection>> IdentifyComplexCodeAsync(string filePath, string language, int threshold = 10);

    /// <summary>
    /// Suggests simplifications for complex code
    /// </summary>
    /// <param name="complexCodeSection">Complex code section</param>
    /// <returns>List of suggested simplifications</returns>
    Task<List<CodeSimplification>> SuggestSimplificationsAsync(ComplexCodeSection complexCodeSection);
}

/// <summary>
/// Represents a complexity analysis result
/// </summary>
public class ComplexityAnalysisResult
{
    /// <summary>
    /// Average cyclomatic complexity
    /// </summary>
    public float AverageCyclomaticComplexity { get; set; }

    /// <summary>
    /// Maximum cyclomatic complexity
    /// </summary>
    public int MaxCyclomaticComplexity { get; set; }

    /// <summary>
    /// Average cognitive complexity
    /// </summary>
    public float AverageCognitiveComplexity { get; set; }

    /// <summary>
    /// Maximum cognitive complexity
    /// </summary>
    public int MaxCognitiveComplexity { get; set; }

    /// <summary>
    /// Average Halstead complexity
    /// </summary>
    public float AverageHalsteadComplexity { get; set; }

    /// <summary>
    /// Maximum Halstead complexity
    /// </summary>
    public float MaxHalsteadComplexity { get; set; }

    /// <summary>
    /// Average maintainability index
    /// </summary>
    public float AverageMaintainabilityIndex { get; set; }

    /// <summary>
    /// Minimum maintainability index
    /// </summary>
    public float MinMaintainabilityIndex { get; set; }

    /// <summary>
    /// List of complex methods
    /// </summary>
    public List<ComplexMethod> ComplexMethods { get; set; } = new();

    /// <summary>
    /// List of complex classes
    /// </summary>
    public List<ComplexClass> ComplexClasses { get; set; } = new();

    /// <summary>
    /// Complexity distribution
    /// </summary>
    public ComplexityDistribution ComplexityDistribution { get; set; } = new();
}

/// <summary>
/// Represents a complex class
/// </summary>
public class ComplexClass
{
    /// <summary>
    /// Class name
    /// </summary>
    public string ClassName { get; set; } = string.Empty;

    /// <summary>
    /// File path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Line number
    /// </summary>
    public int LineNumber { get; set; }

    /// <summary>
    /// Class length in lines
    /// </summary>
    public int ClassLength { get; set; }

    /// <summary>
    /// Number of methods
    /// </summary>
    public int MethodCount { get; set; }

    /// <summary>
    /// Number of properties
    /// </summary>
    public int PropertyCount { get; set; }

    /// <summary>
    /// Number of fields
    /// </summary>
    public int FieldCount { get; set; }

    /// <summary>
    /// Weighted method count
    /// </summary>
    public int WeightedMethodCount { get; set; }

    /// <summary>
    /// Depth of inheritance tree
    /// </summary>
    public int InheritanceDepth { get; set; }

    /// <summary>
    /// Number of children
    /// </summary>
    public int ChildrenCount { get; set; }
}

/// <summary>
/// Represents a complexity distribution
/// </summary>
public class ComplexityDistribution
{
    /// <summary>
    /// Distribution of cyclomatic complexity
    /// </summary>
    public Dictionary<int, int> CyclomaticComplexityDistribution { get; set; } = new();

    /// <summary>
    /// Distribution of cognitive complexity
    /// </summary>
    public Dictionary<int, int> CognitiveComplexityDistribution { get; set; } = new();

    /// <summary>
    /// Distribution of maintainability index
    /// </summary>
    public Dictionary<int, int> MaintainabilityIndexDistribution { get; set; } = new();

    /// <summary>
    /// Distribution of method length
    /// </summary>
    public Dictionary<int, int> MethodLengthDistribution { get; set; } = new();

    /// <summary>
    /// Distribution of class length
    /// </summary>
    public Dictionary<int, int> ClassLengthDistribution { get; set; } = new();
}

/// <summary>
/// Represents a complex code section
/// </summary>
public class ComplexCodeSection
{
    /// <summary>
    /// File path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Start line
    /// </summary>
    public int StartLine { get; set; }

    /// <summary>
    /// End line
    /// </summary>
    public int EndLine { get; set; }

    /// <summary>
    /// Code content
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Complexity type
    /// </summary>
    public UnifiedComplexityType ComplexityType { get; set; } = UnifiedComplexityType.Cyclomatic;

    /// <summary>
    /// Complexity value
    /// </summary>
    public int ComplexityValue { get; set; }

    /// <summary>
    /// Method name (if applicable)
    /// </summary>
    public string? MethodName { get; set; }

    /// <summary>
    /// Class name (if applicable)
    /// </summary>
    public string? ClassName { get; set; }
}

/// <summary>
/// Represents a code simplification
/// </summary>
public class CodeSimplification
{
    /// <summary>
    /// Description of the simplification
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Simplified code
    /// </summary>
    public string SimplifiedCode { get; set; } = string.Empty;

    /// <summary>
    /// Complexity reduction
    /// </summary>
    public int ComplexityReduction { get; set; }

    /// <summary>
    /// Confidence in the simplification (0-1)
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Potential risks of the simplification
    /// </summary>
    public List<string> PotentialRisks { get; set; } = new();
}

/// <summary>
/// Type of complexity
/// </summary>
public enum ComplexityType
{
    /// <summary>
    /// Cyclomatic complexity
    /// </summary>
    Cyclomatic,

    /// <summary>
    /// Cognitive complexity
    /// </summary>
    Cognitive,

    /// <summary>
    /// Halstead complexity
    /// </summary>
    Halstead,

    /// <summary>
    /// Method length
    /// </summary>
    MethodLength,

    /// <summary>
    /// Class length
    /// </summary>
    ClassLength,

    /// <summary>
    /// Parameter count
    /// </summary>
    ParameterCount,

    /// <summary>
    /// Nesting depth
    /// </summary>
    NestingDepth,

    /// <summary>
    /// Maintainability index
    /// </summary>
    Maintainability,

    /// <summary>
    /// Maintainability index (alternative name)
    /// </summary>
    MaintainabilityIndex
}
