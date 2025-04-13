using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for detecting style issues in code
/// </summary>
public interface IStyleIssueDetector : IIssueDetector
{
    /// <summary>
    /// Detects inconsistent naming conventions
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectInconsistentNaming(string content);

    /// <summary>
    /// Detects inconsistent indentation
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectInconsistentIndentation(string content);

    /// <summary>
    /// Detects inconsistent brace style
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectInconsistentBraceStyle(string content);

    /// <summary>
    /// Detects magic numbers
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectMagicNumbers(string content);
}