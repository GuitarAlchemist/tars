using System.Collections.Generic;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces
{
    /// <summary>
    /// Interface for detecting complexity issues in code
    /// </summary>
    public interface IComplexityIssueDetector : IIssueDetector
    {
        /// <summary>
        /// Detects methods with high cyclomatic complexity
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <param name="structures">The extracted code structures</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectHighCyclomaticComplexity(string content, List<CodeStructure> structures);

        /// <summary>
        /// Detects methods with too many parameters
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectTooManyParameters(string content);

        /// <summary>
        /// Detects methods that are too long
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <param name="structures">The extracted code structures</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectMethodsTooLong(string content, List<CodeStructure> structures);

        /// <summary>
        /// Detects deeply nested code
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectDeeplyNestedCode(string content);
    }
}
