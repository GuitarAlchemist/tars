using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for detecting code issues
/// </summary>
public interface IIssueDetector
{
    /// <summary>
    /// Gets the language supported by this detector
    /// </summary>
    string Language { get; }

    /// <summary>
    /// Gets the type of issues detected by this detector
    /// </summary>
    CodeIssueType IssueType { get; }

    /// <summary>
    /// Detects issues in the provided code content
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <param name="structures">The extracted code structures</param>
    /// <returns>A list of detected issues</returns>
    List<CodeIssue> DetectIssues(string content, List<CodeStructure> structures);

    /// <summary>
    /// Gets the available issue severities for this detector
    /// </summary>
    /// <returns>A dictionary of issue severities and their descriptions</returns>
    Dictionary<IssueSeverity, string> GetAvailableSeverities();

    /// <summary>
    /// Gets the line number for a position in the content
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <param name="position">The position in the content</param>
    /// <returns>The line number (1-based)</returns>
    int GetLineNumber(string content, int position);
}