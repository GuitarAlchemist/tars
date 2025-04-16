using System.Text.RegularExpressions;
using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.Adapters;

namespace TarsCli.Services.CodeGeneration;

/// <summary>
/// Generator for F# code
/// </summary>
public class FSharpCodeGenerator : ICodeGenerator
{
    private readonly ILogger<FSharpCodeGenerator> _logger;

    /// <summary>
    /// Initializes a new instance of the FSharpCodeGenerator class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    public FSharpCodeGenerator(ILogger<FSharpCodeGenerator> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc/>
    public async Task<CodeGenerationResult> GenerateCodeAsync(string filePath, string originalContent, CodeAnalysisResult analysisResult)
    {
        _logger.LogInformation($"Generating improved F# code for {filePath}");

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

                        case CodeIssueType.Design:
                            newLines = ImproveDesign(newLines, lineIndex, issue);
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

            _logger.LogInformation($"Generated improved F# code for {filePath} with {result.Changes.Count} changes");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating improved F# code for {filePath}");
            result.ErrorMessage = ex.Message;
            result.GeneratedContent = originalContent;
        }

        return await Task.FromResult(result);
    }

    /// <inheritdoc/>
    public IEnumerable<string> GetSupportedFileExtensions()
    {
        return [".fs", ".fsx"];
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

        // Add parameter documentation for functions
        if (memberType == "Function" && line.Contains("->"))
        {
            var parameters = ExtractParameters(line);
            foreach (var param in parameters)
            {
                docLines.Add($"/// <param name=\"{param}\">The {param} parameter</param>");
            }

            // Add return documentation
            docLines.Add("/// <returns>The result of the function</returns>");
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
                // Convert snake_case to camelCase
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
            if (line.Contains("+") && line.Contains("\""))
            {
                var variableName = line.Split('=')[0].Trim();
                newLines[lineIndex] = $"let sb = new System.Text.StringBuilder({variableName})";
                newLines[lineIndex + 1] = line.Replace("+", "sb.Append(") + ") |> ignore";
                newLines[lineIndex + 2] = $"{variableName} <- sb.ToString()";
            }
        }
        else if (issue.Description.Contains("recursion"))
        {
            // Add tail recursion optimization
            if (line.Contains("let rec"))
            {
                newLines[lineIndex] = line.Replace("let rec", "let rec") + " // TODO: Optimize with tail recursion";
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
            if (line.Contains("sql") && line.Contains("+"))
            {
                // Replace string concatenation with parameterized query
                var paramName = "param" + Guid.NewGuid().ToString("N").Substring(0, 8);
                newLines[lineIndex] = line.Replace("+", "@" + paramName);
                newLines[lineIndex + 1] = $"command.Parameters.AddWithValue(\"@{paramName}\", value) |> ignore";
            }
        }
        else if (issue.Description.Contains("hardcoded credentials"))
        {
            // Fix hardcoded credentials
            if (line.Contains("password") || line.Contains("apikey") || line.Contains("secret"))
            {
                // Replace hardcoded credentials with configuration
                newLines[lineIndex] = $"// TODO: Move credentials to secure configuration";
                newLines[lineIndex + 1] = $"let secureValue = configuration.GetValue<string>(\"SecureConfig:Key\")";
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
                var constantName = $"constant{number}";

                // Add constant declaration at the top of the file
                var newLinesList = newLines.ToList();
                var moduleIndex = FindModuleDeclaration(lines);
                if (moduleIndex >= 0)
                {
                    newLinesList.Insert(moduleIndex + 1, $"let {constantName} = {number}");
                    newLines = newLinesList.ToArray();
                }

                // Replace the number with the constant
                newLines[lineIndex] = line.Replace(number, constantName);
            }
        }
        else if (issue.Description.Contains("unused binding"))
        {
            // Remove unused bindings
            var regex = new Regex(@"let\s+(\w+)\s*=");
            var match = regex.Match(line);
            if (match.Success)
            {
                var bindingName = match.Groups[1].Value;
                newLines[lineIndex] = $"// Removed unused binding: {bindingName}";
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
        if (issue.Description.Contains("long function"))
        {
            // Add a comment suggesting function extraction
            newLines[lineIndex] = $"{line} // TODO: Extract this part into a separate function to reduce complexity";
        }
        else if (issue.Description.Contains("nested if"))
        {
            // Suggest using pattern matching instead of nested if statements
            newLines[lineIndex] = $"{line} // TODO: Replace nested if statements with pattern matching";
        }
        else
        {
            // Generic complexity reduction
            newLines[lineIndex] = $"{line} // TODO: Reduce complexity: {issue.Description}";
        }

        return newLines;
    }

    /// <summary>
    /// Improves code design
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="lineIndex">Index of the line to modify</param>
    /// <param name="issue">Issue to fix</param>
    /// <returns>Modified lines of code</returns>
    private string[] ImproveDesign(string[] lines, int lineIndex, CodeAnalysis.CodeIssue issue)
    {
        var newLines = lines.ToArray();
        var line = lines[lineIndex];

        // Apply design improvements based on the issue description
        if (issue.Description.Contains("mutable"))
        {
            // Replace mutable variables with immutable alternatives
            if (line.Contains("mutable"))
            {
                newLines[lineIndex] = $"{line} // TODO: Replace mutable variable with immutable alternative";
            }
        }
        else if (issue.Description.Contains("imperative"))
        {
            // Replace imperative code with functional alternatives
            if (line.Contains("for") || line.Contains("while"))
            {
                newLines[lineIndex] = $"{line} // TODO: Replace imperative loop with functional alternative (e.g., List.map, List.fold)";
            }
        }
        else
        {
            // Generic design improvement
            newLines[lineIndex] = $"{line} // TODO: Improve design: {issue.Description}";
        }

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
        // Extract function name
        var functionMatch = Regex.Match(line, @"let\s+(?:rec\s+)?([a-zA-Z0-9_]+)");
        if (functionMatch.Success)
        {
            return functionMatch.Groups[1].Value;
        }

        // Extract type name
        var typeMatch = Regex.Match(line, @"type\s+([a-zA-Z0-9_]+)");
        if (typeMatch.Success)
        {
            return typeMatch.Groups[1].Value;
        }

        // Extract module name
        var moduleMatch = Regex.Match(line, @"module\s+([a-zA-Z0-9_\.]+)");
        if (moduleMatch.Success)
        {
            return moduleMatch.Groups[1].Value;
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
        if (line.Contains("type "))
        {
            if (line.Contains("=") && (line.Contains("|") || line.Contains("{")))
            {
                return "Type";
            }
            else
            {
                return "Type declaration";
            }
        }
        else if (line.Contains("module "))
        {
            return "Module";
        }
        else if (line.Contains("let "))
        {
            if (line.Contains("->") || line.Contains("function") || line.Contains("fun"))
            {
                return "Function";
            }
            else
            {
                return "Value";
            }
        }
        else
        {
            return "Member";
        }
    }

    /// <summary>
    /// Extracts parameters from a function declaration
    /// </summary>
    /// <param name="line">Line of code</param>
    /// <returns>List of parameter names</returns>
    private List<string> ExtractParameters(string line)
    {
        var parameters = new List<string>();

        // Extract function parameters
        var match = Regex.Match(line, @"let\s+(?:rec\s+)?[a-zA-Z0-9_]+\s+((?:[a-zA-Z0-9_]+\s+)+)");
        if (match.Success)
        {
            var paramList = match.Groups[1].Value;
            var paramMatches = Regex.Matches(paramList, @"([a-zA-Z0-9_]+)\s+");
            foreach (Match paramMatch in paramMatches)
            {
                parameters.Add(paramMatch.Groups[1].Value);
            }
        }

        return parameters;
    }

    /// <summary>
    /// Finds the module declaration in the code
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <returns>Index of the module declaration, or -1 if not found</returns>
    private int FindModuleDeclaration(string[] lines)
    {
        for (var i = 0; i < lines.Length; i++)
        {
            if (Regex.IsMatch(lines[i], @"module\s+[a-zA-Z0-9_\.]+"))
            {
                return i;
            }
        }
        return -1;
    }
}