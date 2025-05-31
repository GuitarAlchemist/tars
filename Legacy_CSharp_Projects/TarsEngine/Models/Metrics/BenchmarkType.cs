namespace TarsEngine.Models.Metrics;

/// <summary>
/// Represents a type of benchmark
/// </summary>
public enum BenchmarkType
{
    /// <summary>
    /// Code complexity benchmark
    /// </summary>
    CodeComplexity,
        
    /// <summary>
    /// Code maintainability benchmark
    /// </summary>
    CodeMaintainability,
        
    /// <summary>
    /// Problem solving benchmark
    /// </summary>
    ProblemSolving,
        
    /// <summary>
    /// Learning efficiency benchmark
    /// </summary>
    LearningEfficiency,
        
    /// <summary>
    /// Knowledge integration benchmark
    /// </summary>
    KnowledgeIntegration,
        
    /// <summary>
    /// Creativity benchmark
    /// </summary>
    Creativity,
        
    /// <summary>
    /// Adaptation benchmark
    /// </summary>
    Adaptation,
        
    /// <summary>
    /// Performance benchmark
    /// </summary>
    Performance
}
    
/// <summary>
/// Extension methods for <see cref="BenchmarkType"/>
/// </summary>
public static class BenchmarkTypeExtensions
{
    /// <summary>
    /// Gets the display name for a benchmark type
    /// </summary>
    /// <param name="benchmarkType">The benchmark type</param>
    /// <returns>The display name</returns>
    public static string GetDisplayName(this BenchmarkType benchmarkType)
    {
        return benchmarkType switch
        {
            BenchmarkType.CodeComplexity => "Code Complexity",
            BenchmarkType.CodeMaintainability => "Code Maintainability",
            BenchmarkType.ProblemSolving => "Problem Solving",
            BenchmarkType.LearningEfficiency => "Learning Efficiency",
            BenchmarkType.KnowledgeIntegration => "Knowledge Integration",
            BenchmarkType.Creativity => "Creativity",
            BenchmarkType.Adaptation => "Adaptation",
            BenchmarkType.Performance => "Performance",
            _ => benchmarkType.ToString()
        };
    }
        
    /// <summary>
    /// Gets the description for a benchmark type
    /// </summary>
    /// <param name="benchmarkType">The benchmark type</param>
    /// <returns>The description</returns>
    public static string GetDescription(this BenchmarkType benchmarkType)
    {
        return benchmarkType switch
        {
            BenchmarkType.CodeComplexity => "Measures the complexity of code produced by TARS",
            BenchmarkType.CodeMaintainability => "Measures the maintainability of code produced by TARS",
            BenchmarkType.ProblemSolving => "Measures TARS's ability to solve problems",
            BenchmarkType.LearningEfficiency => "Measures how efficiently TARS learns new concepts",
            BenchmarkType.KnowledgeIntegration => "Measures TARS's ability to integrate knowledge from different domains",
            BenchmarkType.Creativity => "Measures TARS's creative capabilities",
            BenchmarkType.Adaptation => "Measures TARS's ability to adapt to new situations",
            BenchmarkType.Performance => "Measures TARS's performance on various tasks",
            _ => "No description available"
        };
    }
}