using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Analyzes code for style issues
/// </summary>
public class StyleAnalyzer(ILogger<StyleAnalyzer> logger)
{
    private readonly ILogger<StyleAnalyzer> _logger = logger;

    /// <summary>
    /// Detects style issues in the provided code content
    /// </summary>
    /// <param name="content">The source code content</param>
    /// <param name="language">The programming language</param>
    /// <returns>A list of detected style issues</returns>
    public List<CodeIssue> DetectStyleIssues(string content, string language)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            issues.AddRange(DetectInconsistentNaming(content, language));
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

    /// <summary>
    /// Detects inconsistent naming conventions
    /// </summary>
    public List<CodeIssue> DetectInconsistentNaming(string content, string language)
    {
        var issues = new List<CodeIssue>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return issues;
            }

            if (language.Equals("csharp", StringComparison.OrdinalIgnoreCase))
            {
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
                                Severity = IssueSeverity.Minor,
                                Title = "Non-Pascal Case Class Name",
                                Description = $"Class name '{className}' does not follow Pascal case convention. Consider renaming to '{ToPascalCase(className)}'.",
                                Location = new CodeLocation
                                {
                                    StartLine = lineNumber,
                                    EndLine = lineNumber
                                }
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
                                Severity = IssueSeverity.Minor,
                                Title = "Non-Pascal Case Method Name",
                                Description = $"Method name '{methodName}' does not follow Pascal case convention. Consider renaming to '{ToPascalCase(methodName)}'.",
                                Location = new CodeLocation
                                {
                                    StartLine = lineNumber,
                                    EndLine = lineNumber
                                }
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
                                Severity = IssueSeverity.Trivial,
                                Title = "Non-Camel Case Variable Name",
                                Description = $"Variable name '{variableName}' does not follow camel case convention. Consider renaming to '{ToCamelCase(variableName)}'.",
                                Location = new CodeLocation
                                {
                                    StartLine = lineNumber,
                                    EndLine = lineNumber
                                }
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
                                Severity = IssueSeverity.Trivial,
                                Title = "Non-Underscore Prefixed Private Field",
                                Description = $"Private field '{fieldName}' does not follow underscore prefix convention. Consider renaming to '_{fieldName}'.",
                                Location = new CodeLocation
                                {
                                    StartLine = lineNumber,
                                    EndLine = lineNumber
                                }
                            });
                        }
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

    /// <summary>
    /// Detects inconsistent indentation
    /// </summary>
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
            var expectedIndent = 0;
            var lineNumber = 1;
                
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
                var actualIndent = line.Length - trimmedLine.Length;
                if (actualIndent != expectedIndent && !trimmedLine.StartsWith("#"))
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Style,
                        Severity = IssueSeverity.Trivial,
                        Title = "Inconsistent Indentation",
                        Description = $"Inconsistent indentation (expected {expectedIndent} spaces, got {actualIndent}). Indent with {expectedIndent} spaces.",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        }
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

    /// <summary>
    /// Detects inconsistent brace style
    /// </summary>
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
            for (var i = 0; i < lines.Length - 1; i++)
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
                    var j = i + 1;
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
                                Severity = IssueSeverity.Trivial,
                                Title = "Inconsistent Brace Style",
                                Description = "Inconsistent brace style (Allman style detected). Use K&R style (opening brace on same line as control statement).",
                                Location = new CodeLocation
                                {
                                    StartLine = j + 2,
                                    EndLine = j + 2
                                }
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
                        Severity = IssueSeverity.Trivial,
                        Title = "Inconsistent Brace Style",
                        Description = "Inconsistent brace style (Allman style detected). Use K&R style (opening brace on same line as control statement).",
                        Location = new CodeLocation
                        {
                            StartLine = i + 2,
                            EndLine = i + 2
                        }
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

    /// <summary>
    /// Detects magic numbers
    /// </summary>
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
                        Severity = IssueSeverity.Trivial,
                        Title = "Magic Number",
                        Description = $"Magic number '{number}' detected. Replace with a named constant.",
                        Location = new CodeLocation
                        {
                            StartLine = lineNumber,
                            EndLine = lineNumber
                        }
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
    /// Gets the available issue severities for style issues
    /// </summary>
    public Dictionary<IssueSeverity, string> GetAvailableSeverities()
    {
        return new Dictionary<IssueSeverity, string>
        {
            { IssueSeverity.Critical, "Critical style issue that violates team standards" },
            { IssueSeverity.Major, "High-impact style issue that affects readability significantly" },
            { IssueSeverity.Minor, "Medium-impact style issue" },
            { IssueSeverity.Trivial, "Low-impact style issue" },
            { IssueSeverity.Info, "Informational style suggestion" }
        };
    }

    /// <summary>
    /// Checks if a string is in Pascal case
    /// </summary>
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
    private static string ToPascalCase(string s)
    {
        if (string.IsNullOrEmpty(s))
        {
            return s;
        }

        // Split by underscores and non-alphanumeric characters
        var parts = Regex.Split(s, @"[^a-zA-Z0-9]");
            
        // Convert each part to Pascal case
        for (var i = 0; i < parts.Length; i++)
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
    private static string GetContext(string content, int position, int length)
    {
        if (string.IsNullOrEmpty(content) || position < 0 || position >= content.Length)
        {
            return string.Empty;
        }

        var start = Math.Max(0, position - length);
        var end = Math.Min(content.Length, position + length);
            
        return content.Substring(start, end - start);
    }
}