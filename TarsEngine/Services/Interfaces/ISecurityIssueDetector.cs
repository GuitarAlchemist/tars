using System.Collections.Generic;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces
{
    /// <summary>
    /// Interface for detecting security issues in code
    /// </summary>
    public interface ISecurityIssueDetector : IIssueDetector
    {
        /// <summary>
        /// Detects SQL injection vulnerabilities
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectSqlInjectionVulnerabilities(string content);

        /// <summary>
        /// Detects XSS vulnerabilities
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectXssVulnerabilities(string content);

        /// <summary>
        /// Detects hardcoded credentials
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectHardcodedCredentials(string content);

        /// <summary>
        /// Detects insecure cryptography
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <returns>A list of detected issues</returns>
        List<CodeIssue> DetectInsecureCryptography(string content);
    }
}
