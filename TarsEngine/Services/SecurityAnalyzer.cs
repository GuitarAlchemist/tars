using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Analyzes code for security vulnerabilities
/// </summary>
public class SecurityAnalyzer(ILogger<SecurityAnalyzer> logger)
{
    private readonly ILogger<SecurityAnalyzer> _logger = logger;

    /// <summary>
    /// Detects security issues in the provided code content
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <param name="language">The programming language</param>
    /// <returns>A list of detected security issues</returns>
    public List<CodeIssue> DetectSecurityIssues(string content, string language)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Detect SQL injection vulnerabilities
            var sqlInjectionRegex = new Regex(@"SqlCommand\s*\(\s*[""'].*?\+\s*[^""']+\s*\+", RegexOptions.Compiled);
            var sqlInjectionMatches = sqlInjectionRegex.Matches(content);
            foreach (Match match in sqlInjectionMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = IssueSeverity.Critical,
                    Title = "SQL Injection Vulnerability",
                    Description = "Potential SQL injection vulnerability detected. String concatenation in SQL queries can lead to SQL injection attacks.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect string concatenation in SQL queries
            var stringConcatRegex = new Regex(@"string\s+sql\s*=\s*[""'].*?\+\s*[^""']+\s*\+", RegexOptions.Compiled);
            var stringConcatMatches = stringConcatRegex.Matches(content);
            foreach (Match match in stringConcatMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = IssueSeverity.Major,
                    Title = "SQL Injection Risk",
                    Description = "String concatenation in SQL query could lead to SQL injection. Use parameterized queries or an ORM instead.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect XSS vulnerabilities
            var xssRegex = new Regex(@"Response\.Write\s*\(\s*[^""']*\s*\)", RegexOptions.Compiled);
            var xssMatches = xssRegex.Matches(content);
            foreach (Match match in xssMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = IssueSeverity.Critical,
                    Title = "XSS Vulnerability",
                    Description = "Potential XSS vulnerability detected. Unencoded output can lead to cross-site scripting attacks.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect unencoded output
            var unencodedRegex = new Regex(@"@Html\.Raw\s*\(\s*[^""']*\s*\)", RegexOptions.Compiled);
            var unencodedMatches = unencodedRegex.Matches(content);
            foreach (Match match in unencodedMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = IssueSeverity.Major,
                    Title = "Unencoded Output",
                    Description = "Unencoded output could lead to XSS vulnerability. Use @Html.Encode() or @Html.DisplayFor() instead of @Html.Raw().",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect hardcoded credentials
            var credentialsRegex = new Regex(@"(password|pwd|passwd|secret|key|token|apikey)\s*=\s*[""'][^""']+[""']", RegexOptions.IgnoreCase | RegexOptions.Compiled);
            var credentialsMatches = credentialsRegex.Matches(content);
            foreach (Match match in credentialsMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.SecurityHotspot,
                    Severity = IssueSeverity.Major,
                    Title = "Hardcoded Credentials",
                    Description = "Hardcoded credentials detected. Store credentials in a secure configuration system or environment variables.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect connection strings with credentials
            var connectionStringRegex = new Regex(@"connectionString\s*=\s*[""'].*?(password|pwd)=[^;]+[""']", RegexOptions.IgnoreCase | RegexOptions.Compiled);
            var connectionStringMatches = connectionStringRegex.Matches(content);
            foreach (Match match in connectionStringMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.SecurityHotspot,
                    Severity = IssueSeverity.Major,
                    Title = "Connection String with Credentials",
                    Description = "Connection string with hardcoded credentials detected. Store connection strings in a secure configuration system.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect weak hash algorithms
            var weakHashRegex = new Regex(@"(MD5|SHA1)\.Create\(\)", RegexOptions.Compiled);
            var weakHashMatches = weakHashRegex.Matches(content);
            foreach (Match match in weakHashMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = IssueSeverity.Major,
                    Title = "Weak Hash Algorithm",
                    Description = "Weak hash algorithm detected. Use a stronger hash algorithm like SHA256 or SHA512.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect weak encryption algorithms
            var weakEncryptionRegex = new Regex(@"(DES|TripleDES|RC2)\.Create\(\)", RegexOptions.Compiled);
            var weakEncryptionMatches = weakEncryptionRegex.Matches(content);
            foreach (Match match in weakEncryptionMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Vulnerability,
                    Severity = IssueSeverity.Major,
                    Title = "Weak Encryption Algorithm",
                    Description = "Weak encryption algorithm detected. Use a stronger encryption algorithm like AES.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }

            // Detect insecure random number generation
            var insecureRandomRegex = new Regex(@"new\s+Random\s*\(\s*\)", RegexOptions.Compiled);
            var insecureRandomMatches = insecureRandomRegex.Matches(content);
            foreach (Match match in insecureRandomMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.SecurityHotspot,
                    Severity = IssueSeverity.Minor,
                    Title = "Insecure Random Number Generation",
                    Description = "Insecure random number generation detected. Use RNGCryptoServiceProvider or RandomNumberGenerator for cryptographic operations.",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    }
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting security issues in code");
        }

        return issues;
    }

    /// <summary>
    /// Gets the line number for a position in the content
    /// </summary>
    public int GetLineNumber(string content, int position)
    {
        if (string.IsNullOrEmpty(content) || position < 0 || position >= content.Length)
        {
            return 0;
        }

        // Count newlines before the position
        return content[..position].Count(c => c == '\n') + 1;
    }

    /// <summary>
    /// Gets the available issue severities for security issues
    /// </summary>
    public Dictionary<IssueSeverity, string> GetAvailableSeverities()
    {
        return new Dictionary<IssueSeverity, string>
        {
            { IssueSeverity.Critical, "Critical security vulnerability that must be fixed immediately" },
            { IssueSeverity.Major, "High-risk security vulnerability that should be fixed soon" },
            { IssueSeverity.Minor, "Medium-risk security vulnerability" },
            { IssueSeverity.Trivial, "Low-risk security vulnerability" },
            { IssueSeverity.Info, "Informational security issue" }
        };
    }
}