using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services
{
    /// <summary>
    /// Detects performance issues in code
    /// </summary>
    public class PerformanceIssueDetector(ILogger<PerformanceIssueDetector> logger) : IPerformanceIssueDetector
    {
        private readonly ILogger<PerformanceIssueDetector> _logger = logger;

        /// <inheritdoc/>
        public string Language => "csharp";

        /// <inheritdoc/>
        public CodeIssueType IssueType => CodeIssueType.Performance;

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

                issues.AddRange(DetectInefficientLoops(content));
                issues.AddRange(DetectLargeObjectCreation(content));
                issues.AddRange(DetectExcessiveMemoryUsage(content));
                issues.AddRange(DetectInefficientStringOperations(content));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting performance issues in code");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectInefficientLoops(string content)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    return issues;
                }

                // Detect collection modification inside foreach loops
                var foreachModificationRegex = new Regex(@"foreach\s*\([^)]+\)\s*{[^}]*\b(Add|Remove|Clear|Insert)\b", RegexOptions.Compiled);
                var foreachModificationMatches = foreachModificationRegex.Matches(content);
                foreach (Match match in foreachModificationMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Major,
                        Description = "Use a for loop or ToList() when modifying collections during iteration",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        Title = "Collection Modification in Loop"
                    });
                }

                // Detect inefficient LINQ usage
                var inefficientLinqRegex = new Regex(@"\.Where\([^)]+\)\.Where\(", RegexOptions.Compiled);
                var inefficientLinqMatches = inefficientLinqRegex.Matches(content);
                foreach (Match match in inefficientLinqMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = "Combine multiple Where() calls into a single call with a compound condition",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        Title = "Multiple Where() Calls"
                    });
                }

                // Detect inefficient loop variable usage
                var loopVariableRegex = new Regex(@"for\s*\([^;]+;\s*[^;]+;\s*[^)]+\)\s*{[^}]*\b(Length|Count)\b", RegexOptions.Compiled);
                var loopVariableMatches = loopVariableRegex.Matches(content);
                foreach (Match match in loopVariableMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Trivial,
                        Description = "Cache the collection length/count in a variable before the loop",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        Title = "Uncached Collection Length"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting inefficient loops");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectLargeObjectCreation(string content)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    return issues;
                }

                // Detect large array creation
                var largeArrayRegex = new Regex(@"new\s+[a-zA-Z0-9_<>]+\s*\[\s*(\d+)\s*\]", RegexOptions.Compiled);
                var largeArrayMatches = largeArrayRegex.Matches(content);
                foreach (Match match in largeArrayMatches)
                {
                    if (match.Groups.Count > 1 && int.TryParse(match.Groups[1].Value, out int size) && size > 1000000)
                    {
                        var lineNumber = GetLineNumber(content, match.Index);
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Performance,
                            Severity = TarsEngine.Models.IssueSeverity.Major,
                            Description = $"Consider using a more memory-efficient data structure or streaming approach for large array (size: {size})",
                            Location = new CodeLocation
                            {
                                StartLine = lineNumber,
                                EndLine = lineNumber
                            },
                            Title = "Large Array Creation"
                        });
                    }
                }

                // Detect large collection initializers
                var largeCollectionRegex = new Regex(@"new\s+[a-zA-Z0-9_<>]+\s*\{([^}]{1000,})\}", RegexOptions.Compiled);
                var largeCollectionMatches = largeCollectionRegex.Matches(content);
                foreach (Match match in largeCollectionMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = "Consider initializing the collection incrementally or using a factory method",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        Title = "Large Collection Initializer"
                    });
                }

                // Detect large string concatenation
                var largeStringConcatRegex = new Regex(@"string\s+[a-zA-Z0-9_]+\s*=\s*([^;]+\+[^;]+){5,}", RegexOptions.Compiled);
                var largeStringConcatMatches = largeStringConcatRegex.Matches(content);
                foreach (Match match in largeStringConcatMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = "Use StringBuilder for multiple string concatenations",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        Title = "Multiple String Concatenations"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting large object creation");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectExcessiveMemoryUsage(string content)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    return issues;
                }

                // Detect missing using statements for disposable objects
                var disposableRegex = new Regex(@"new\s+(FileStream|StreamReader|StreamWriter|SqlConnection|SqlCommand|MemoryStream|NetworkStream)\s*\([^)]*\)(?!\s*using|\s*=\s*null|\s*;\s*using|\s*;\s*try)", RegexOptions.Compiled);
                var disposableMatches = disposableRegex.Matches(content);
                foreach (Match match in disposableMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Critical,
                        Description = "Disposable object created without using statement or proper disposal",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = match.Value,
                        SuggestedFix = "Use a using statement or using declaration to ensure proper disposal"
                    });
                }

                // Detect large static collections
                var staticCollectionRegex = new Regex(@"static\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*=\s*new\s+[a-zA-Z0-9_<>]+", RegexOptions.Compiled);
                var staticCollectionMatches = staticCollectionRegex.Matches(content);
                foreach (Match match in staticCollectionMatches)
                {
                    if (match.Value.Contains("List<") || match.Value.Contains("Dictionary<") ||
                        match.Value.Contains("HashSet<") || match.Value.Contains("Collection<"))
                    {
                        var lineNumber = GetLineNumber(content, match.Index);
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Performance,
                            Severity = TarsEngine.Models.IssueSeverity.Major,
                            Description = "Static collection detected",
                            Location = new CodeLocation
                            {
                                StartLine = lineNumber,
                                EndLine = lineNumber
                            },
                            CodeSnippet = match.Value,
                            SuggestedFix = "Consider using a non-static collection or implementing a cache with size limits"
                        });
                    }
                }

                // Detect excessive boxing/unboxing
                var boxingRegex = new Regex(@"(object)\s+[a-zA-Z0-9_]+\s*=\s*\d+|object\s*\[\s*\]|ArrayList|Hashtable", RegexOptions.Compiled);
                var boxingMatches = boxingRegex.Matches(content);
                foreach (Match match in boxingMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Major,
                        Description = "Potential boxing/unboxing operation detected",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = match.Value,
                        SuggestedFix = "Use generic collections and avoid unnecessary conversions between value types and object"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting excessive memory usage");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectInefficientStringOperations(string content)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    return issues;
                }

                // Detect string concatenation in loops
                var stringConcatLoopRegex = new Regex(@"(for|foreach|while)[^{]*{[^}]*\+\=\s*[""']", RegexOptions.Compiled);
                var stringConcatLoopMatches = stringConcatLoopRegex.Matches(content);
                foreach (Match match in stringConcatLoopMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Critical,
                        Description = "String concatenation in loop detected",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = match.Value,
                        SuggestedFix = "Use StringBuilder for string concatenation in loops"
                    });
                }

                // Detect inefficient string.Substring usage
                var substringRegex = new Regex(@"\.Substring\s*\(\s*0\s*,", RegexOptions.Compiled);
                var substringMatches = substringRegex.Matches(content);
                foreach (Match match in substringMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = "Inefficient Substring usage detected",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = match.Value,
                        SuggestedFix = "Use string[..end] range operator instead of Substring(0, end)"
                    });
                }

                // Detect repeated string.Replace calls
                var replaceRegex = new Regex(@"\.Replace\s*\([^)]+\)\.Replace\s*\(", RegexOptions.Compiled);
                var replaceMatches = replaceRegex.Matches(content);
                foreach (Match match in replaceMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Major,
                        Description = "Multiple string.Replace calls in chain detected",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = match.Value,
                        SuggestedFix = "Consider using Regex.Replace for multiple replacements or a StringBuilder"
                    });
                }

                // Detect inefficient string comparison
                var comparisonRegex = new Regex(@"\.ToLower\s*\(\s*\)\s*==|\.ToUpper\s*\(\s*\)\s*==", RegexOptions.Compiled);
                var comparisonMatches = comparisonRegex.Matches(content);
                foreach (Match match in comparisonMatches)
                {
                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Performance,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = "Case-insensitive string comparison using ToLower()/ToUpper() detected",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = match.Value,
                        SuggestedFix = "Use string.Equals(s1, s2, StringComparison.OrdinalIgnoreCase) for case-insensitive comparison"
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting inefficient string operations");
            }

            return issues;
        }

        /// <inheritdoc/>
        public Dictionary<TarsEngine.Services.Interfaces.IssueSeverity, string> GetAvailableSeverities()
        {
            return new Dictionary<TarsEngine.Services.Interfaces.IssueSeverity, string>
            {
                { TarsEngine.Services.Interfaces.IssueSeverity.Critical, "Critical performance issue that causes significant slowdowns" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Major, "High-impact performance issue that should be fixed soon" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Minor, "Medium-impact performance issue" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Trivial, "Low-impact performance issue" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Warning, "Informational performance suggestion" }
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
}
