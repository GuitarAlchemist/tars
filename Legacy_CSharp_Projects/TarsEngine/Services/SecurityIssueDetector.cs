using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Detects security issues in code
/// </summary>
public class SecurityIssueDetector(ILogger<SecurityIssueDetector> logger) : ISecurityIssueDetector
{
    private readonly ILogger<SecurityIssueDetector> _logger = logger;

    /// <inheritdoc/>
    public string Language => "csharp";

    /// <inheritdoc/>
    public CodeIssueType IssueType => CodeIssueType.Security;

    /// <inheritdoc/>
    public List<CodeIssue> DetectIssues(string content, List<CodeStructure> structures)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            issues.AddRange(DetectSqlInjectionVulnerabilities(content));
            issues.AddRange(DetectXssVulnerabilities(content));
            issues.AddRange(DetectHardcodedCredentials(content));
            issues.AddRange(DetectInsecureCryptography(content));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting security issues in code");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectSqlInjectionVulnerabilities(string content)
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
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Description = "Use parameterized queries or an ORM instead of string concatenation",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "SQL Injection Vulnerability"
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
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Major,
                    Description = "Use parameterized queries or an ORM instead of string concatenation",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "SQL Injection Risk"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting SQL injection vulnerabilities");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectXssVulnerabilities(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Detect XSS vulnerabilities
            var xssRegex = new Regex(@"Response\.Write\s*\(\s*[^""']*\s*\)", RegexOptions.Compiled);
            var xssMatches = xssRegex.Matches(content);
            foreach (Match match in xssMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Critical,
                    Description = "Use HTML encoding or a templating engine that automatically escapes output",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "XSS Vulnerability"
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
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Major,
                    Description = "Use @Html.Encode() or @Html.DisplayFor() instead of @Html.Raw()",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "Unencoded Output"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting XSS vulnerabilities");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectHardcodedCredentials(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Detect hardcoded credentials
            var credentialsRegex = new Regex(@"(password|pwd|passwd|secret|key|token|apikey)\s*=\s*[""'][^""']+[""']", RegexOptions.IgnoreCase | RegexOptions.Compiled);
            var credentialsMatches = credentialsRegex.Matches(content);
            foreach (Match match in credentialsMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Major,
                    Description = "Store credentials in a secure configuration system or environment variables",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "Hardcoded Credentials"
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
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Major,
                    Description = "Store connection strings in a secure configuration system or use a connection string builder with secure parameters",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "Connection String with Credentials"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting hardcoded credentials");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectInsecureCryptography(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Detect weak hash algorithms
            var weakHashRegex = new Regex(@"(MD5|SHA1)\.Create\(\)", RegexOptions.Compiled);
            var weakHashMatches = weakHashRegex.Matches(content);
            foreach (Match match in weakHashMatches)
            {
                var lineNumber = GetLineNumber(content, match.Index);
                issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Major,
                    Description = "Use a stronger hash algorithm like SHA256 or SHA512",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "Weak Hash Algorithm"
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
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Major,
                    Description = "Use a stronger encryption algorithm like AES",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "Weak Encryption Algorithm"
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
                    Type = CodeIssueType.Security,
                    Severity = TarsEngine.Models.IssueSeverity.Minor,
                    Description = "Use RNGCryptoServiceProvider or RandomNumberGenerator for cryptographic operations",
                    Location = new CodeLocation
                    {
                        StartLine = lineNumber,
                        EndLine = lineNumber
                    },
                    Title = "Insecure Random Number Generation"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting insecure cryptography");
        }

        return issues;
    }

    /// <inheritdoc/>
    public Dictionary<Interfaces.IssueSeverity, string> GetAvailableSeverities()
    {
        return new Dictionary<Interfaces.IssueSeverity, string>
        {
            { Interfaces.IssueSeverity.Critical, "Critical security vulnerability that must be fixed immediately" },
            { Interfaces.IssueSeverity.Major, "High-risk security vulnerability that should be fixed soon" },
            { Interfaces.IssueSeverity.Minor, "Medium-risk security vulnerability" },
            { Interfaces.IssueSeverity.Trivial, "Low-risk security vulnerability" },
            { Interfaces.IssueSeverity.Warning, "Informational security issue" }
        };
    }

    /// <inheritdoc/>
    public int GetLineNumber(string content, int position)
    {
        if (string.IsNullOrEmpty(content) || position < 0 || position >= content.Length)
        {
            return 0;
        }

        // Count newlines before the position
        return content[..position].Count(c => c == '\n') + 1;
    }
}