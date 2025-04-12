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
    /// Detects complexity issues in code
    /// </summary>
    public class ComplexityIssueDetector(ILogger<ComplexityIssueDetector> logger) : IComplexityIssueDetector
    {
        private readonly ILogger<ComplexityIssueDetector> _logger = logger;

        /// <inheritdoc/>
        public string Language => "csharp";

        /// <inheritdoc/>
        public CodeIssueType IssueType => CodeIssueType.Complexity;

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

                issues.AddRange(DetectHighCyclomaticComplexity(content, structures));
                issues.AddRange(DetectTooManyParameters(content));
                issues.AddRange(DetectMethodsTooLong(content, structures));
                issues.AddRange(DetectDeeplyNestedCode(content));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting complexity issues in code");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectHighCyclomaticComplexity(string content, List<CodeStructure> structures)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content) || structures == null || !structures.Any())
                {
                    return issues;
                }

                // Get method structures
                var methodStructures = structures.Where(s => s.Type == StructureType.Method).ToList();

                foreach (var method in methodStructures)
                {
                    // Get method content
                    var methodContent = GetMethodContent(content, method);
                    if (string.IsNullOrWhiteSpace(methodContent))
                    {
                        continue;
                    }

                    // Calculate cyclomatic complexity
                    int complexity = CalculateCyclomaticComplexity(methodContent);

                    // Add issue if complexity is too high
                    if (complexity > 10)
                    {
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Complexity,
                            Severity = complexity > 15 ? TarsEngine.Models.IssueSeverity.Critical : TarsEngine.Models.IssueSeverity.Major,
                            Description = $"Method has high cyclomatic complexity ({complexity})",
                            Location = new CodeLocation
                            {
                                StartLine = method.Location.StartLine,
                                EndLine = method.Location.EndLine,
                                FilePath = method.Location.FilePath,
                                Namespace = method.Location.Namespace,
                                ClassName = method.Location.ClassName
                            },
                            Code = method.Name,
                            SuggestedFix = "Refactor the method into smaller, more focused methods"
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting high cyclomatic complexity");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectTooManyParameters(string content)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    return issues;
                }

                // Detect methods with too many parameters
                var methodRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(([^)]+)\)", RegexOptions.Compiled);
                var methodMatches = methodRegex.Matches(content);
                foreach (Match match in methodMatches)
                {
                    if (match.Groups.Count > 5)
                    {
                        var methodName = match.Groups[4].Value;
                        var parameters = match.Groups[5].Value;
                        var parameterCount = parameters.Split(',').Length;

                        if (parameterCount > 5)
                        {
                            var lineNumber = GetLineNumber(content, match.Index);
                            issues.Add(new CodeIssue
                            {
                                Type = CodeIssueType.Complexity,
                                Severity = parameterCount > 8 ? TarsEngine.Models.IssueSeverity.Critical : TarsEngine.Models.IssueSeverity.Major,
                                Description = $"Method has too many parameters ({parameterCount})",
                                Location = new CodeLocation
                                {
                                    StartLine = lineNumber,
                                    EndLine = lineNumber
                                },
                                Code = methodName,
                                SuggestedFix = "Consider using parameter objects or breaking the method into smaller methods"
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting methods with too many parameters");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectMethodsTooLong(string content, List<CodeStructure> structures)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content) || structures == null || !structures.Any())
                {
                    return issues;
                }

                // Get method structures
                var methodStructures = structures.Where(s => s.Type == StructureType.Method).ToList();

                foreach (var method in methodStructures)
                {
                    // Check if method is too long
                    if (method.Size > 30)
                    {
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Complexity,
                            Severity = method.Size > 50 ? TarsEngine.Models.IssueSeverity.Critical : TarsEngine.Models.IssueSeverity.Major,
                            Description = $"Method is too long ({method.Size} lines)",
                            Location = new CodeLocation
                            {
                                StartLine = method.Location.StartLine,
                                EndLine = method.Location.EndLine,
                                FilePath = method.Location.FilePath,
                                Namespace = method.Location.Namespace,
                                ClassName = method.Location.ClassName
                            },
                            Code = method.Name,
                            SuggestedFix = "Break the method into smaller, more focused methods"
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting methods that are too long");
            }

            return issues;
        }

        /// <inheritdoc/>
        public List<CodeIssue> DetectDeeplyNestedCode(string content)
        {
            var issues = new List<CodeIssue>();

            try
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    return issues;
                }

                // Find all opening braces
                var bracePositions = new List<int>();
                for (int i = 0; i < content.Length; i++)
                {
                    if (content[i] == '{')
                    {
                        bracePositions.Add(i);
                    }
                }

                // Check nesting level at each brace position
                foreach (var position in bracePositions)
                {
                    int nestingLevel = 0;
                    for (int i = 0; i < position; i++)
                    {
                        if (content[i] == '{')
                        {
                            nestingLevel++;
                        }
                        else if (content[i] == '}')
                        {
                            nestingLevel--;
                        }
                    }

                    // If nesting level is too deep, add an issue
                    if (nestingLevel > 3)
                    {
                        var lineNumber = GetLineNumber(content, position);

                        // Get some context around the position
                        int contextStart = Math.Max(0, position - 50);
                        int contextEnd = Math.Min(content.Length, position + 50);
                        string contextCode = content.Substring(contextStart, contextEnd - contextStart);

                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Complexity,
                            Severity = nestingLevel > 4 ? TarsEngine.Models.IssueSeverity.Critical : TarsEngine.Models.IssueSeverity.Major,
                            Description = $"Deeply nested code detected (level {nestingLevel})",
                            Location = new CodeLocation
                            {
                                StartLine = lineNumber,
                                EndLine = lineNumber
                            },
                            Code = contextCode,
                            SuggestedFix = "Refactor to reduce nesting by extracting methods or using early returns"
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting deeply nested code");
            }

            return issues;
        }

        /// <inheritdoc/>
        public Dictionary<TarsEngine.Services.Interfaces.IssueSeverity, string> GetAvailableSeverities()
        {
            return new Dictionary<TarsEngine.Services.Interfaces.IssueSeverity, string>
            {
                { TarsEngine.Services.Interfaces.IssueSeverity.Critical, "Critical complexity issue that makes code unmaintainable" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Major, "High-impact complexity issue that should be fixed soon" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Minor, "Medium-impact complexity issue" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Trivial, "Low-impact complexity issue" },
                { TarsEngine.Services.Interfaces.IssueSeverity.Warning, "Informational complexity suggestion" }
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

        /// <summary>
        /// Calculates the cyclomatic complexity of a method
        /// </summary>
        /// <param name="methodContent">The method content</param>
        /// <returns>The cyclomatic complexity value</returns>
        public int CalculateCyclomaticComplexity(string methodContent)
        {
            if (string.IsNullOrWhiteSpace(methodContent))
            {
                return 1;
            }

            // Start with 1 (base complexity)
            int complexity = 1;

            // Count decision points
            complexity += Regex.Matches(methodContent, @"\bif\b").Count;
            complexity += Regex.Matches(methodContent, @"\belse\s+if\b").Count;
            complexity += Regex.Matches(methodContent, @"\bwhile\b").Count;
            complexity += Regex.Matches(methodContent, @"\bfor\b").Count;
            complexity += Regex.Matches(methodContent, @"\bforeach\b").Count;
            complexity += Regex.Matches(methodContent, @"\bcase\b").Count;
            complexity += Regex.Matches(methodContent, @"\bcatch\b").Count;
            complexity += Regex.Matches(methodContent, @"\b\|\|\b").Count;
            complexity += Regex.Matches(methodContent, @"\b&&\b").Count;
            complexity += Regex.Matches(methodContent, @"\?\s*[^:]+\s*:").Count; // Ternary operators

            return complexity;
        }

        /// <summary>
        /// Gets the content of a method from the full code content
        /// </summary>
        /// <param name="content">The full code content</param>
        /// <param name="method">The method structure</param>
        /// <returns>The method content</returns>
        private string GetMethodContent(string content, CodeStructure method)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(content) || method == null)
                {
                    return string.Empty;
                }

                // Find the method declaration
                var methodRegex = new Regex($@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*[a-zA-Z0-9_<>]+\s+{Regex.Escape(method.Name)}\s*\(", RegexOptions.Compiled);
                var match = methodRegex.Match(content);
                if (!match.Success)
                {
                    return string.Empty;
                }

                // Find the opening brace
                int openBracePos = content.IndexOf('{', match.Index);
                if (openBracePos == -1)
                {
                    return string.Empty;
                }

                // Find the matching closing brace
                int closeBracePos = FindMatchingBrace(content, openBracePos);
                if (closeBracePos == -1)
                {
                    return string.Empty;
                }

                // Extract the method content
                return content.Substring(match.Index, closeBracePos - match.Index + 1);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting method content");
                return string.Empty;
            }
        }

        /// <summary>
        /// Finds the position of the matching closing brace
        /// </summary>
        /// <param name="content">The content to search in</param>
        /// <param name="openBracePos">The position of the opening brace</param>
        /// <returns>The position of the matching closing brace, or -1 if not found</returns>
        private static int FindMatchingBrace(string content, int openBracePos)
        {
            int braceCount = 1;
            for (int i = openBracePos + 1; i < content.Length; i++)
            {
                if (content[i] == '{')
                {
                    braceCount++;
                }
                else if (content[i] == '}')
                {
                    braceCount--;
                    if (braceCount == 0)
                    {
                        return i;
                    }
                }
            }
            return -1;
        }
    }
}
