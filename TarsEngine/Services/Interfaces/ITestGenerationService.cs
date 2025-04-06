using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the test generation service
/// </summary>
public interface ITestGenerationService
{
    /// <summary>
    /// Generates tests for a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to generate tests for</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <param name="testFramework">Test framework to use (e.g., xUnit, NUnit)</param>
    /// <returns>Generated test code</returns>
    Task<string> GenerateTestsForFileAsync(string filePath, string projectPath, string testFramework = "xUnit");

    /// <summary>
    /// Generates tests for a specific method
    /// </summary>
    /// <param name="filePath">Path to the file containing the method</param>
    /// <param name="methodName">Name of the method to generate tests for</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <param name="testFramework">Test framework to use (e.g., xUnit, NUnit)</param>
    /// <returns>Generated test code</returns>
    Task<string> GenerateTestsForMethodAsync(string filePath, string methodName, string projectPath, string testFramework = "xUnit");

    /// <summary>
    /// Generates tests for improved code
    /// </summary>
    /// <param name="originalCode">Original code</param>
    /// <param name="improvedCode">Improved code</param>
    /// <param name="language">Programming language</param>
    /// <param name="testFramework">Test framework to use (e.g., xUnit, NUnit)</param>
    /// <returns>Generated test code</returns>
    Task<string> GenerateTestsForImprovedCodeAsync(string originalCode, string improvedCode, string language, string testFramework = "xUnit");

    /// <summary>
    /// Suggests test cases based on code analysis
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <returns>List of suggested test cases</returns>
    Task<List<TestCase>> SuggestTestCasesAsync(string filePath, string projectPath);
}

/// <summary>
/// Represents a test case
/// </summary>
public class TestCase
{
    /// <summary>
    /// Name of the test case
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Description of the test case
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Input values for the test case
    /// </summary>
    public Dictionary<string, object> Inputs { get; set; } = new();

    /// <summary>
    /// Expected output for the test case
    /// </summary>
    public object? ExpectedOutput { get; set; }

    /// <summary>
    /// Expected exception for the test case (if any)
    /// </summary>
    public string? ExpectedException { get; set; }

    /// <summary>
    /// Priority of the test case
    /// </summary>
    public TestCasePriority Priority { get; set; } = TestCasePriority.Medium;

    /// <summary>
    /// Category of the test case
    /// </summary>
    public string Category { get; set; } = string.Empty;
}

/// <summary>
/// Priority of a test case
/// </summary>
public enum TestCasePriority
{
    Low,
    Medium,
    High,
    Critical
}
