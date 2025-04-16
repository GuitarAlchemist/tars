using System.Text;
using System.Text.RegularExpressions;
using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.Adapters;

namespace TarsCli.Services.CodeGeneration;

/// <summary>
/// Generator for C# code
/// </summary>
public class CSharpCodeGenerator : ICodeGenerator
{
    private readonly ILogger<CSharpCodeGenerator> _logger;

    /// <summary>
    /// Initializes a new instance of the CSharpCodeGenerator class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    public CSharpCodeGenerator(ILogger<CSharpCodeGenerator> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc/>
    public async Task<CodeGenerationResult> GenerateCodeAsync(string filePath, string originalContent, CodeAnalysisResult analysisResult)
    {
        _logger.LogInformation($"Generating improved C# code for {filePath}");

        var result = new CodeGenerationResult
        {
            FilePath = filePath,
            OriginalContent = originalContent,
            Success = false
        };

        try
        {
            // Split the file content into lines for processing
            var lines = originalContent.Split(["\r\n", "\r", "\n"], StringSplitOptions.None);
            var newLines = new string[lines.Length];
            Array.Copy(lines, newLines, lines.Length);

            // Group issues by line number
            var issuesByLine = analysisResult.Issues
                .GroupBy(i => i.LineNumber)
                .ToDictionary(g => g.Key, g => g.ToList());

            // Process each issue
            foreach (var lineNumber in issuesByLine.Keys.OrderByDescending(ln => ln))
            {
                var lineIssues = issuesByLine[lineNumber];
                var lineIndex = lineNumber - 1;

                if (lineIndex < 0 || lineIndex >= lines.Length)
                {
                    continue;
                }

                foreach (var issue in lineIssues.OrderByDescending(i => i.Severity))
                {
                    // Apply the appropriate fix based on the issue type
                    switch (issue.Type)
                    {
                        case CodeIssueType.Documentation:
                            newLines = AddDocumentation(newLines, lineIndex, issue);
                            break;

                        case CodeIssueType.Style:
                            newLines = FixStyleIssue(newLines, lineIndex, issue);
                            break;

                        case CodeIssueType.Performance:
                            newLines = OptimizeCode(newLines, lineIndex, issue);
                            break;

                        case CodeIssueType.Security:
                            newLines = FixSecurityIssue(newLines, lineIndex, issue);
                            break;

                        case CodeIssueType.Maintainability:
                            newLines = ImproveMaintenability(newLines, lineIndex, issue);
                            break;

                        case CodeIssueType.Complexity:
                            newLines = ReduceComplexity(newLines, lineIndex, issue);
                            break;

                        case CodeIssueType.Duplication:
                            newLines = RemoveDuplication(newLines, lineIndex, issue);
                            break;

                        default:
                            // For other issue types, just add a comment
                            newLines[lineIndex] = $"{newLines[lineIndex]} // TODO: {issue.Description}";
                            break;
                    }

                    // Record the change
                    result.Changes.Add(new CodeChange
                    {
                        Type = GetChangeType(issue.Type),
                        Description = issue.Description,
                        LineNumber = lineNumber,
                        OriginalCode = lines[lineIndex],
                        NewCode = newLines[lineIndex],
                        Issue = CodeIssueAdapter.ToServiceCodeIssue(issue)
                    });
                }
            }

            // Combine the lines back into a single string
            result.GeneratedContent = string.Join(Environment.NewLine, newLines);
            result.Success = true;

            _logger.LogInformation($"Generated improved C# code for {filePath} with {result.Changes.Count} changes");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating improved C# code for {filePath}");
            result.ErrorMessage = ex.Message;
            result.GeneratedContent = originalContent;
        }

        return await Task.FromResult(result);
    }

    /// <inheritdoc/>
    public IEnumerable<string> GetSupportedFileExtensions()
    {
        return [".cs"];
    }

    /// <summary>
    /// Adds documentation to a line
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] AddDocumentation(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = new List<string>(lines);
        var line = lines[lineIndex];

        // Extract the member name
        var memberName = ExtractMemberName(line);
        var memberType = DetermineMemberType(line);

        // Create the documentation
        var docLines = new List<string>
        {
            "/// <summary>",
            $"/// {memberType} {memberName}",
            "/// </summary>"
        };

        // Add parameter documentation for methods
        if (memberType == "Method" && line.Contains("("))
        {
            var parameters = ExtractParameters(line);
            foreach (var param in parameters)
            {
                docLines.Add($"/// <param name=\"{param}\">The {param} parameter</param>");
            }

            // Add return documentation if not void
            if (!line.Contains("void "))
            {
                docLines.Add("/// <returns>The result of the operation</returns>");
            }
        }

        // Insert the documentation before the line
        newLines.InsertRange(lineIndex, docLines);

        return newLines.ToArray();
    }

    /// <summary>
    /// Fixes a style issue
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] FixStyleIssue(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = lines.ToArray();
        var line = lines[lineIndex];

        // Apply style fixes based on the issue description
        if (issue.Description.Contains("naming"))
        {
            // Fix naming issues
            if (line.Contains("_"))
            {
                // Convert snake_case to camelCase or PascalCase
                var regex = new Regex(@"(\b|_)([a-z])");
                newLines[lineIndex] = regex.Replace(line, m => m.Groups[2].Value.ToUpper());
            }
        }
        else if (issue.Description.Contains("spacing"))
        {
            // Fix spacing issues
            newLines[lineIndex] = Regex.Replace(line, @"\s+", " ");
            newLines[lineIndex] = Regex.Replace(newLines[lineIndex], @"\(\s+", "(");
            newLines[lineIndex] = Regex.Replace(newLines[lineIndex], @"\s+\)", ")");
            newLines[lineIndex] = Regex.Replace(newLines[lineIndex], @"\{\s+", "{ ");
            newLines[lineIndex] = Regex.Replace(newLines[lineIndex], @"\s+\}", " }");
        }
        else if (issue.Description.Contains("braces"))
        {
            // Fix braces issues
            if (line.Contains("if") || line.Contains("for") || line.Contains("while") || line.Contains("foreach"))
            {
                if (!line.Contains("{"))
                {
                    newLines[lineIndex] = line + " {";
                    // Add closing brace on the next line
                    var indentation = line.Length - line.TrimStart().Length;
                    var closingBrace = new string(' ', indentation) + "}";
                    var newLinesList = newLines.ToList();
                    newLinesList.Insert(lineIndex + 2, closingBrace);
                    newLines = newLinesList.ToArray();
                }
            }
        }
        else
        {
            // Generic style fix
            newLines[lineIndex] = $"{line} // TODO: Fix style issue: {issue.Description}";
        }

        return newLines;
    }

    /// <summary>
    /// Optimizes code for performance
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] OptimizeCode(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = lines.ToArray();
        var line = lines[lineIndex];

        // Apply performance optimizations based on the issue description
        if (issue.Description.Contains("string concatenation"))
        {
            // Replace string concatenation with StringBuilder
            if (line.Contains("+=") && line.Contains("\""))
            {
                var variableName = line.Split('=')[0].Trim();
                newLines[lineIndex] = $"var sb = new StringBuilder({variableName});";
                newLines[lineIndex + 1] = line.Replace("+=", "sb.Append(") + ");";
                newLines[lineIndex + 2] = $"{variableName} = sb.ToString();";
            }
        }
        else if (issue.Description.Contains("LINQ"))
        {
            // Optimize LINQ queries
            if (line.Contains(".Where(") && line.Contains(".Select(") && line.Contains(".ToList()"))
            {
                // Combine Where and Select
                var newLine = line.Replace(".Where(", ".Select(").Replace(".Select(", ".Where(");
                newLines[lineIndex] = newLine;
            }
        }
        else
        {
            // Generic performance optimization
            newLines[lineIndex] = $"{line} // TODO: Optimize for performance: {issue.Description}";
        }

        return newLines;
    }

    /// <summary>
    /// Fixes a security issue
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] FixSecurityIssue(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = lines.ToArray();
        var line = lines[lineIndex];

        // Apply security fixes based on the issue description
        if (issue.Description.Contains("SQL injection"))
        {
            // Fix SQL injection vulnerabilities
            if (line.Contains("string sql") && line.Contains("+"))
            {
                // Replace string concatenation with parameterized query
                var paramName = "param" + Guid.NewGuid().ToString("N").Substring(0, 8);
                newLines[lineIndex] = line.Replace("+", "@" + paramName + ";");
                newLines[lineIndex + 1] = $"command.Parameters.AddWithValue(\"@{paramName}\", value);";
            }
        }
        else if (issue.Description.Contains("XSS"))
        {
            // Fix XSS vulnerabilities
            if (line.Contains("Response.Write"))
            {
                // Add HTML encoding
                newLines[lineIndex] = line.Replace("Response.Write(", "Response.Write(HttpUtility.HtmlEncode(");
            }
        }
        else if (issue.Description.Contains("hardcoded credentials"))
        {
            // Fix hardcoded credentials
            if (line.Contains("password") || line.Contains("apikey") || line.Contains("secret"))
            {
                // Replace hardcoded credentials with configuration
                newLines[lineIndex] = $"// TODO: Move credentials to secure configuration";
                newLines[lineIndex + 1] = $"var secureValue = Configuration.GetValue<string>(\"SecureConfig:Key\");";
            }
        }
        else
        {
            // Generic security fix
            newLines[lineIndex] = $"{line} // TODO: Fix security issue: {issue.Description}";
        }

        return newLines;
    }

    /// <summary>
    /// Improves code maintainability
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] ImproveMaintenability(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = lines.ToArray();
        var line = lines[lineIndex];

        // Apply maintainability improvements based on the issue description
        if (issue.Description.Contains("magic number"))
        {
            // Replace magic numbers with constants
            var regex = new Regex(@"(\d+)");
            var match = regex.Match(line);
            if (match.Success)
            {
                var number = match.Groups[1].Value;
                var constantName = $"CONSTANT_{number}";

                // Add constant declaration at the top of the file
                var newLinesList = newLines.ToList();
                var classIndex = FindClassDeclaration(lines);
                if (classIndex >= 0)
                {
                    newLinesList.Insert(classIndex + 1, $"    private const int {constantName} = {number};");
                    newLines = newLinesList.ToArray();
                }

                // Replace the number with the constant
                newLines[lineIndex] = line.Replace(number, constantName);
            }
        }
        else if (issue.Description.Contains("unused variable"))
        {
            // Remove unused variables
            var regex = new Regex(@"(var|int|string|bool|double|float|decimal|long|short|byte|char|object|dynamic)\s+(\w+)\s*=");
            var match = regex.Match(line);
            if (match.Success)
            {
                var variableName = match.Groups[2].Value;
                newLines[lineIndex] = $"// Removed unused variable: {variableName}";
            }
        }
        else
        {
            // Generic maintainability improvement
            newLines[lineIndex] = $"{line} // TODO: Improve maintainability: {issue.Description}";
        }

        return newLines;
    }

    /// <summary>
    /// Reduces code complexity
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] ReduceComplexity(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = lines.ToArray();
        var line = lines[lineIndex];

        // Apply complexity reductions based on the issue description
        if (issue.Description.Contains("complex conditional"))
        {
            // Extract complex conditionals into separate methods
            var methodName = $"Check{Guid.NewGuid().ToString("N").Substring(0, 8)}";

            // Add new method at the end of the file
            var newLinesList = newLines.ToList();
            var lastBraceIndex = FindLastClosingBrace(lines);
            if (lastBraceIndex >= 0)
            {
                var indentation = line.Length - line.TrimStart().Length;
                var spaces = new string(' ', indentation);

                newLinesList.Insert(lastBraceIndex, $"{spaces}private bool {methodName}()");
                newLinesList.Insert(lastBraceIndex + 1, $"{spaces}{{");
                newLinesList.Insert(lastBraceIndex + 2, $"{spaces}    return {line.Trim()};");
                newLinesList.Insert(lastBraceIndex + 3, $"{spaces}}}");

                newLines = newLinesList.ToArray();
            }

            // Replace the complex conditional with a method call
            newLines[lineIndex] = line.Replace(line.Trim(), $"{methodName}()");
        }
        else if (issue.Description.Contains("long method"))
        {
            // Add a comment suggesting method extraction
            newLines[lineIndex] = $"{line} // TODO: Extract this part into a separate method to reduce complexity";
        }
        else
        {
            // Generic complexity reduction
            newLines[lineIndex] = $"{line} // TODO: Reduce complexity: {issue.Description}";
        }

        return newLines;
    }

    /// <summary>
    /// Removes code duplication
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] RemoveDuplication(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = lines.ToArray();
        var line = lines[lineIndex];

        // Add a comment suggesting duplication removal
        newLines[lineIndex] = $"{line} // TODO: Extract this duplicated code into a shared method";

        return newLines;
    }

    /// <summary>
    /// Gets the change type based on the issue type
    /// </summary>
    /// <param name="issueType">Issue type</param>
    /// <returns>Change type</returns>
    private CodeChangeType GetChangeType(CodeIssueType issueType)
    {
        switch (issueType)
        {
            case CodeIssueType.Documentation:
                return CodeChangeType.Documentation;
            case CodeIssueType.Style:
                return CodeChangeType.StyleFix;
            case CodeIssueType.Performance:
                return CodeChangeType.Optimization;
            case CodeIssueType.Security:
                return CodeChangeType.SecurityFix;
            case CodeIssueType.Maintainability:
            case CodeIssueType.Reliability:
                return CodeChangeType.Modification;
            case CodeIssueType.Complexity:
            case CodeIssueType.Design:
                return CodeChangeType.Refactoring;
            case CodeIssueType.Duplication:
                return CodeChangeType.Refactoring;
            default:
                return CodeChangeType.Modification;
        }
    }

    /// <summary>
    /// Extracts the member name from a line of code
    /// </summary>
    /// <param name="line">Line of code</param>
    /// <returns>Member name</returns>
    private string ExtractMemberName(string line)
    {
        // Extract method name
        var methodMatch = Regex.Match(line, @"(?:public|private|protected|internal)\s+(?:static\s+)?(?:virtual\s+)?(?:override\s+)?(?:async\s+)?(?:[a-zA-Z0-9_<>]+\s+)+([a-zA-Z0-9_]+)\s*\(");
        if (methodMatch.Success)
        {
            return methodMatch.Groups[1].Value;
        }

        // Extract property name
        var propertyMatch = Regex.Match(line, @"(?:public|private|protected|internal)\s+(?:static\s+)?(?:virtual\s+)?(?:override\s+)?(?:[a-zA-Z0-9_<>]+\s+)+([a-zA-Z0-9_]+)\s*\{");
        if (propertyMatch.Success)
        {
            return propertyMatch.Groups[1].Value;
        }

        // Extract class name
        var classMatch = Regex.Match(line, @"(?:public|private|protected|internal)\s+(?:static\s+)?(?:abstract\s+)?class\s+([a-zA-Z0-9_]+)");
        if (classMatch.Success)
        {
            return classMatch.Groups[1].Value;
        }

        // Extract interface name
        var interfaceMatch = Regex.Match(line, @"(?:public|private|protected|internal)\s+interface\s+([a-zA-Z0-9_]+)");
        if (interfaceMatch.Success)
        {
            return interfaceMatch.Groups[1].Value;
        }

        // Extract enum name
        var enumMatch = Regex.Match(line, @"(?:public|private|protected|internal)\s+enum\s+([a-zA-Z0-9_]+)");
        if (enumMatch.Success)
        {
            return enumMatch.Groups[1].Value;
        }

        // Extract field name
        var fieldMatch = Regex.Match(line, @"(?:public|private|protected|internal)\s+(?:static\s+)?(?:readonly\s+)?(?:[a-zA-Z0-9_<>]+\s+)+([a-zA-Z0-9_]+)\s*=");
        if (fieldMatch.Success)
        {
            return fieldMatch.Groups[1].Value;
        }

        // Default
        return "Unknown";
    }

    /// <summary>
    /// Determines the member type from a line of code
    /// </summary>
    /// <param name="line">Line of code</param>
    /// <returns>Member type</returns>
    private string DetermineMemberType(string line)
    {
        if (line.Contains("class "))
        {
            return "Class";
        }
        else if (line.Contains("interface "))
        {
            return "Interface";
        }
        else if (line.Contains("enum "))
        {
            return "Enum";
        }
        else if (line.Contains("("))
        {
            return "Method";
        }
        else if (line.Contains("{") && line.Contains("get") || line.Contains("set"))
        {
            return "Property";
        }
        else
        {
            return "Field";
        }
    }

    /// <summary>
    /// Extracts parameters from a method declaration
    /// </summary>
    /// <param name="line">Line of code</param>
    /// <returns>List of parameter names</returns>
    private List<string> ExtractParameters(string line)
    {
        var parameters = new List<string>();

        // Extract the parameter list
        var match = Regex.Match(line, @"\((.*)\)");
        if (match.Success)
        {
            var paramList = match.Groups[1].Value;
            if (!string.IsNullOrWhiteSpace(paramList))
            {
                // Split by commas, but handle generic types correctly
                var paramParts = new List<string>();
                var currentPart = new StringBuilder();
                var angleBracketCount = 0;

                foreach (var c in paramList)
                {
                    if (c == '<')
                    {
                        angleBracketCount++;
                        currentPart.Append(c);
                    }
                    else if (c == '>')
                    {
                        angleBracketCount--;
                        currentPart.Append(c);
                    }
                    else if (c == ',' && angleBracketCount == 0)
                    {
                        paramParts.Add(currentPart.ToString());
                        currentPart.Clear();
                    }
                    else
                    {
                        currentPart.Append(c);
                    }
                }

                if (currentPart.Length > 0)
                {
                    paramParts.Add(currentPart.ToString());
                }

                // Extract parameter names
                foreach (var part in paramParts)
                {
                    var paramMatch = Regex.Match(part.Trim(), @"(?:[a-zA-Z0-9_<>]+\s+)+([a-zA-Z0-9_]+)(?:\s*=.*)?$");
                    if (paramMatch.Success)
                    {
                        parameters.Add(paramMatch.Groups[1].Value);
                    }
                }
            }
        }

        return parameters;
    }

    /// <summary>
    /// Finds the class declaration in the code
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <returns>Index of the class declaration, or -1 if not found</returns>
    private int FindClassDeclaration(string[] lines)
    {
        for (var i = 0; i < lines.Length; i++)
        {
            if (Regex.IsMatch(lines[i], @"(?:public|private|protected|internal)\s+(?:static\s+)?(?:abstract\s+)?class\s+[a-zA-Z0-9_]+"))
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>
    /// Finds the last closing brace in the code
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <returns>Index of the last closing brace, or -1 if not found</returns>
    private int FindLastClosingBrace(string[] lines)
    {
        for (var i = lines.Length - 1; i >= 0; i--)
        {
            if (lines[i].Trim() == "}")
            {
                return i;
            }
        }
        return -1;
    }
}