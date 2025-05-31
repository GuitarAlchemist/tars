using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for detecting performance issues in code
/// </summary>
public interface IPerformanceIssueDetector : IIssueDetector
{
    /// <summary>
    /// Detects inefficient loops
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectInefficientLoops(string content);

    /// <summary>
    /// Detects large object creation
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectLargeObjectCreation(string content);

    /// <summary>
    /// Detects excessive memory usage
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectExcessiveMemoryUsage(string content);

    /// <summary>
    /// Detects inefficient string operations
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectInefficientStringOperations(string content);
}