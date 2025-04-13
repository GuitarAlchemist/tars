using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Detects style issues in code
/// </summary>
public class StyleIssueDetector(ILogger<StyleIssueDetector> logger) : IStyleIssueDetector
{
    private readonly ILogger<StyleIssueDetector> _logger = logger;

    /// <inheritdoc/>
    public string Language => "csharp";

    /// <inheritdoc/>
    public CodeIssueType IssueType => CodeIssueType.Style;

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

            issues.AddRange(DetectInconsistentNaming(content));
            issues.AddRange(DetectInconsistentIndentation(content));
            issues.AddRange(DetectInconsistentBraceStyle(content));
            issues.AddRange(DetectMagicNumbers(content));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting style issues in code");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectInconsistentNaming(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Detect non-Pascal case class names
            var classRegex = new Regex(@"class\s+([a-zA-Z0-9_]+)", RegexOptions.Compiled);
            var classMatches = classRegex.Matches(content);
            foreach (Match match in classMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var className = match.Groups[1].Value;
                    if (!IsPascalCase(className))
                    {
                        var lineNumber = GetLineNumber(content, match.Index);
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Style,
                            Severity = TarsEngine.Models.IssueSeverity.Major,
                            Description = "Class name does not follow Pascal case convention",
                            Location = new CodeLocation
                            {
                                StartLine = lineNumber,
                                EndLine = lineNumber
                            },
                            CodeSnippet = className,
                            SuggestedFix = $"Rename to {ToPascalCase(className)}"
                        });
                    }
                }
            }

            // Detect non-Pascal case method names
            var methodRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*[a-zA-Z0-9_<>]+\s+([a-zA-Z0-9_]+)\s*\(", RegexOptions.Compiled);
            var methodMatches = methodRegex.Matches(content);
            foreach (Match match in methodMatches)
            {
                if (match.Groups.Count > 3)
                {
                    var methodName = match.Groups[3].Value;
                    if (!IsPascalCase(methodName))
                    {
                        var lineNumber = GetLineNumber(content, match.Index);
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Style,
                            Severity = TarsEngine.Models.IssueSeverity.Major,
                            Description = "Method name does not follow Pascal case convention",
                            Location = new CodeLocation
                            {
                                StartLine = lineNumber,
                                EndLine = lineNumber
                            },
                            CodeSnippet = methodName,
                            SuggestedFix = $"Rename to {ToPascalCase(methodName)}"
                        });
                    }
                }
            }

            // Detect non-camel case variable names
            var variableRegex = new Regex(@"(var|int|string|bool|double|float|decimal|char|byte|short|long|object|dynamic)\s+([a-zA-Z0-9_]+)\s*[=;]", RegexOptions.Compiled);
            var variableMatches = variableRegex.Matches(content);
            foreach (Match match in variableMatches)
            {
                if (match.Groups.Count > 2)
                {
                    var variableName = match.Groups[2].Value;
                    if (!IsCamelCase(variableName) && !variableName.StartsWith("_"))
                    {
                        var lineNumber = GetLineNumber(content, match.Index);
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Style,
                            Severity = TarsEngine.Models.IssueSeverity.Minor,
                            Description = "Variable name does not follow camel case convention",
                            Location = new CodeLocation
                            {
                                StartLine = lineNumber,
                                EndLine = lineNumber
                            },
                            CodeSnippet = variableName,
                            SuggestedFix = $"Rename to {ToCamelCase(variableName)}"
                        });
                    }
                }
            }

            // Detect non-underscore prefixed private fields
            var privateFieldRegex = new Regex(@"private\s+[a-zA-Z0-9_<>]+\s+([a-zA-Z0-9_]+)\s*[=;]", RegexOptions.Compiled);
            var privateFieldMatches = privateFieldRegex.Matches(content);
            foreach (Match match in privateFieldMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var fieldName = match.Groups[1].Value;
                    if (!fieldName.StartsWith("_"))
                    {
                        var lineNumber = GetLineNumber(content, match.Index);
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Style,
                            Severity = TarsEngine.Models.IssueSeverity.Minor,
                            Description = "Private field does not follow underscore prefix convention",
                            Location = new CodeLocation
                            {
                                StartLine = lineNumber,
                                EndLine = lineNumber
                            },
                            CodeSnippet = fieldName,
                            SuggestedFix = $"Rename to _{fieldName}"
                        });
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting inconsistent naming");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectInconsistentIndentation(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Split content into lines
            var lines = content.Split('\n');

            // Check for inconsistent indentation
            int expectedIndent = 0;
            int lineNumber = 1;

            foreach (var line in lines)
            {
                var trimmedLine = line.TrimStart();

                // Skip empty lines and comments
                if (string.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("//") || trimmedLine.StartsWith("/*"))
                {
                    lineNumber++;
                    continue;
                }

                // Check if line starts with closing brace
                if (trimmedLine.StartsWith("}"))
                {
                    expectedIndent = Math.Max(0, expectedIndent - 4);
                }

                // Check indentation
                int actualIndent = line.Length - trimmedLine.Length;
                if (actualIndent != expectedIndent && !trimmedLine.StartsWith("#"))
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Style,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = $"Inconsistent indentation (expected {expectedIndent} spaces, got {actualIndent})",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = line,
                        SuggestedFix = $"Indent with {expectedIndent} spaces"
                    });
                }

                // Check if line ends with opening brace
                if (trimmedLine.EndsWith("{"))
                {
                    expectedIndent += 4;
                }

                lineNumber++;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting inconsistent indentation");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectInconsistentBraceStyle(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Split content into lines
            var lines = content.Split('\n');

            // Check for inconsistent brace style
            for (int i = 0; i < lines.Length - 1; i++)
            {
                var currentLine = lines[i].TrimEnd();
                var nextLine = lines[i + 1].TrimStart();

                // Check for K&R style (brace on same line)
                if ((currentLine.EndsWith("if (") || currentLine.EndsWith("else if (") ||
                     currentLine.EndsWith("for (") || currentLine.EndsWith("foreach (") ||
                     currentLine.EndsWith("while (") || currentLine.EndsWith("do") ||
                     currentLine.EndsWith("switch (")) &&
                    !currentLine.EndsWith("{"))
                {
                    // Find the closing parenthesis
                    int j = i + 1;
                    while (j < lines.Length && !lines[j].Contains(")"))
                    {
                        j++;
                    }

                    if (j < lines.Length)
                    {
                        var lineWithClosingParen = lines[j].TrimEnd();

                        // Check if the next line has the opening brace
                        if (j + 1 < lines.Length && lines[j + 1].TrimStart().StartsWith("{"))
                        {
                            issues.Add(new CodeIssue
                            {
                                Type = CodeIssueType.Style,
                                Severity = TarsEngine.Models.IssueSeverity.Minor,
                                Description = "Inconsistent brace style (Allman style detected)",
                                Location = new CodeLocation
                                {
                                    StartLine = j + 2,
                                    EndLine = j + 2
                                },
                                CodeSnippet = lines[j + 1],
                                SuggestedFix = "Use K&R style (opening brace on same line as control statement)"
                            });
                        }
                    }
                }

                // Check for Allman style (brace on new line)
                if (currentLine.EndsWith(")") && nextLine.StartsWith("{"))
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Style,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = "Inconsistent brace style (Allman style detected)",
                        Location = new CodeLocation
                        {
                            StartLine = i + 2,
                            EndLine = i + 2
                        },
                        CodeSnippet = nextLine,
                        SuggestedFix = "Use K&R style (opening brace on same line as control statement)"
                    });
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting inconsistent brace style");
        }

        return issues;
    }

    /// <inheritdoc/>
    public List<CodeIssue> DetectMagicNumbers(string content)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            // Detect magic numbers
            var magicNumberRegex = new Regex(@"[^a-zA-Z0-9_](\d+)[^a-zA-Z0-9_]", RegexOptions.Compiled);
            var magicNumberMatches = magicNumberRegex.Matches(content);

            foreach (Match match in magicNumberMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var number = match.Groups[1].Value;

                    // Skip common numbers like 0, 1, 2, -1
                    if (number == "0" || number == "1" || number == "2" || number == "-1" ||
                        number == "100" || number == "10" || number == "24" || number == "60" ||
                        number == "365" || number == "12" || number == "7" || number == "30" ||
                        number == "31")
                    {
                        continue;
                    }

                    // Skip numbers in array initializers
                    var context = GetContext(content, match.Index, 20);
                    if (context.Contains("new") && context.Contains("[") && context.Contains("]"))
                    {
                        continue;
                    }

                    // Skip numbers in enum declarations
                    if (context.Contains("enum") && context.Contains("{"))
                    {
                        continue;
                    }

                    var lineNumber = GetLineNumber(content, match.Index);
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Style,
                        Severity = TarsEngine.Models.IssueSeverity.Minor,
                        Description = "Magic number detected",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        },
                        CodeSnippet = number,
                        SuggestedFix = "Replace with a named constant"
                    });
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting magic numbers");
        }

        return issues;
    }

    /// <inheritdoc/>
    public Dictionary<TarsEngine.Services.Interfaces.IssueSeverity, string> GetAvailableSeverities()
    {
        return new Dictionary<TarsEngine.Services.Interfaces.IssueSeverity, string>
        {
            { TarsEngine.Services.Interfaces.IssueSeverity.Critical, "Critical style issue that violates team standards" },
            { TarsEngine.Services.Interfaces.IssueSeverity.Major, "High-impact style issue that affects readability significantly" },
            { TarsEngine.Services.Interfaces.IssueSeverity.Minor, "Medium-impact style issue" },
            { TarsEngine.Services.Interfaces.IssueSeverity.Trivial, "Low-impact style issue" },
            { TarsEngine.Services.Interfaces.IssueSeverity.Warning, "Informational style suggestion" }
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
    /// Checks if a string is in Pascal case
    /// </summary>
    /// <param name="s">The string to check</param>
    /// <returns>True if the string is in Pascal case, false otherwise</returns>
    private static bool IsPascalCase(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return false;
        }

        // First character must be uppercase
        if (!char.IsUpper(s[0]))
        {
            return false;
        }

        // Must not contain underscores
        if (s.Contains('_'))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Checks if a string is in camel case
    /// </summary>
    /// <param name="s">The string to check</param>
    /// <returns>True if the string is in camel case, false otherwise</returns>
    private static bool IsCamelCase(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return false;
        }

        // First character must be lowercase
        if (!char.IsLower(s[0]))
        {
            return false;
        }

        // Must not contain underscores
        if (s.Contains('_'))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Converts a string to Pascal case
    /// </summary>
    /// <param name="s">The string to convert</param>
    /// <returns>The string in Pascal case</returns>
    private static string ToPascalCase(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return s;
        }

        // Split by underscores and non-alphanumeric characters
        var parts = Regex.Split(s, @"[^a-zA-Z0-9]");

        // Convert each part to Pascal case
        for (int i = 0; i < parts.Length; i++)
        {
            if (!string.IsNullOrEmpty(parts[i]))
            {
                parts[i] = char.ToUpper(parts[i][0]) + (parts[i].Length > 1 ? parts[i][1..].ToLower() : "");
            }
        }

        return string.Join("", parts);
    }

    /// <summary>
    /// Converts a string to camel case
    /// </summary>
    /// <param name="s">The string to convert</param>
    /// <returns>The string in camel case</returns>
    private static string ToCamelCase(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return s;
        }

        // Remove leading underscores
        s = s.TrimStart('_');

        // Convert to Pascal case first
        s = ToPascalCase(s);

        // Convert first character to lowercase
        return char.ToLower(s[0]) + s[1..];
    }

    /// <summary>
    /// Gets the context around a position in the content
    /// </summary>
    /// <param name="content">The content</param>
    /// <param name="position">The position</param>
    /// <param name="length">The length of context to get</param>
    /// <returns>The context</returns>
    private static string GetContext(string content, int position, int length)
    {
        if (string.IsNullOrEmpty(content) || position < 0 || position >= content.Length)
        {
            return string.Empty;
        }

        int start = Math.Max(0, position - length);
        int end = Math.Min(content.Length, position + length);

        return content.Substring(start, end - start);
    }
}