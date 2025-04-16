using System.Text.RegularExpressions;

namespace TarsCli.Services.CodeAnalysis;

/// <summary>
/// Analyzer for C# code
/// </summary>
public class CSharpCodeAnalyzer : ICodeAnalyzer
{
    private readonly ILogger<CSharpCodeAnalyzer> _logger;
    private readonly SecurityVulnerabilityAnalyzer _securityAnalyzer;

    /// <summary>
    /// Initializes a new instance of the CSharpCodeAnalyzer class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="securityAnalyzer">Security vulnerability analyzer</param>
    public CSharpCodeAnalyzer(
        ILogger<CSharpCodeAnalyzer> logger,
        SecurityVulnerabilityAnalyzer securityAnalyzer)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _securityAnalyzer = securityAnalyzer ?? throw new ArgumentNullException(nameof(securityAnalyzer));
    }

    /// <inheritdoc/>
    public async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath, string fileContent)
    {
        _logger.LogInformation($"Analyzing C# file: {filePath}");

        var result = new CodeAnalysisResult
        {
            FilePath = filePath,
            NeedsImprovement = false
        };

        try
        {
            // Split the file content into lines for analysis
            var lines = fileContent.Split(["\r\n", "\r", "\n"], StringSplitOptions.None);

            // Calculate basic metrics
            result.Metrics["LineCount"] = lines.Length;
            result.Metrics["EmptyLineCount"] = lines.Count(line => string.IsNullOrWhiteSpace(line));
            result.Metrics["CommentLineCount"] = lines.Count(line => line.Trim().StartsWith("//") || line.Trim().StartsWith("/*") || line.Trim().StartsWith("*"));
            result.Metrics["CodeLineCount"] = result.Metrics["LineCount"] - result.Metrics["EmptyLineCount"] - result.Metrics["CommentLineCount"];

            // Check for missing XML documentation on public members
            await CheckMissingDocumentationAsync(lines, result);

            // Check for unused variables
            await CheckUnusedVariablesAsync(lines, result);

            // Check for long methods
            await CheckLongMethodsAsync(lines, result);

            // Check for complex conditionals
            await CheckComplexConditionalsAsync(lines, result);

            // Check for magic numbers
            await CheckMagicNumbersAsync(lines, result);

            // Check for security vulnerabilities
            var securityIssues = await _securityAnalyzer.AnalyzeAsync(filePath, fileContent, "cs");
            result.Issues.AddRange(securityIssues);

            // Set needs improvement flag if any issues were found
            result.NeedsImprovement = result.Issues.Count > 0;

            _logger.LogInformation($"Analysis completed for {filePath}. Found {result.Issues.Count} issues.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing file {filePath}");
            result.Issues.Add(new CodeIssue
            {
                Type = CodeIssueType.Functional,
                Severity = IssueSeverity.Error,
                Description = $"Error analyzing file: {ex.Message}",
                LineNumber = 1,
                ColumnNumber = 1
            });
        }

        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<string> GetSupportedFileExtensions()
    {
        return [".cs"];
    }

    /// <summary>
    /// Checks for missing XML documentation on public members
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="result">Analysis result</param>
    private async Task CheckMissingDocumentationAsync(string[] lines, CodeAnalysisResult result)
    {
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();

            // Check for public class, method, property, etc. without XML documentation
            if (line.StartsWith("public ") || line.StartsWith("public virtual ") || line.StartsWith("public override "))
            {
                // Check if the previous line has XML documentation
                var hasDocumentation = false;
                for (var j = i - 1; j >= 0 && j >= i - 5; j--)
                {
                    if (lines[j].Trim().StartsWith("///"))
                    {
                        hasDocumentation = true;
                        break;
                    }
                    if (!string.IsNullOrWhiteSpace(lines[j]) && !lines[j].Trim().StartsWith("["))
                    {
                        break;
                    }
                }

                if (!hasDocumentation)
                {
                    result.Issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Documentation,
                        Severity = IssueSeverity.Warning,
                        Description = "Missing XML documentation for public member",
                        LineNumber = i + 1,
                        ColumnNumber = 1,
                        CodeSegment = lines[i],
                        SuggestedFix = $"/// <summary>\n/// Description of this member\n/// </summary>\n{lines[i]}"
                    });
                }
            }
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Checks for unused variables
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="result">Analysis result</param>
    private async Task CheckUnusedVariablesAsync(string[] lines, CodeAnalysisResult result)
    {
        // Simple regex to find variable declarations
        var varDeclarationRegex = new Regex(@"(?:var|int|string|bool|double|float|decimal|long|short|byte|char|object|dynamic)\s+(\w+)\s*=");

        // Find all variable declarations
        var variables = new Dictionary<string, int>();
        for (var i = 0; i < lines.Length; i++)
        {
            var matches = varDeclarationRegex.Matches(lines[i]);
            foreach (Match match in matches)
            {
                if (match.Groups.Count > 1)
                {
                    var varName = match.Groups[1].Value;
                    variables[varName] = i;
                }
            }
        }

        // Check if each variable is used elsewhere in the code
        foreach (var variable in variables)
        {
            var varName = variable.Key;
            var lineNumber = variable.Value;

            var isUsed = false;
            for (var i = 0; i < lines.Length; i++)
            {
                if (i == lineNumber)
                {
                    continue; // Skip the declaration line
                }

                // Simple check for variable usage (this is a simplification)
                if (lines[i].Contains(varName))
                {
                    isUsed = true;
                    break;
                }
            }

            if (!isUsed)
            {
                result.Issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Maintainability,
                    Severity = IssueSeverity.Warning,
                    Description = $"Unused variable: {varName}",
                    LineNumber = lineNumber + 1,
                    ColumnNumber = lines[lineNumber].IndexOf(varName) + 1,
                    CodeSegment = lines[lineNumber],
                    SuggestedFix = $"// Remove unused variable: {varName}"
                });
            }
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Checks for long methods
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="result">Analysis result</param>
    private async Task CheckLongMethodsAsync(string[] lines, CodeAnalysisResult result)
    {
        const int MaxMethodLength = 50; // Maximum acceptable method length

        var methodStartLine = -1;
        var methodName = string.Empty;
        var braceCount = 0;

        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();

            // Check for method declaration
            if (methodStartLine == -1 && (line.Contains(" void ") || line.Contains(" async ") || Regex.IsMatch(line, @"\w+\s+\w+\s*\(")))
            {
                if (line.Contains("(") && !line.Contains("=>"))
                {
                    methodStartLine = i;
                    var match = Regex.Match(line, @"\s(\w+)\s*\(");
                    if (match.Success)
                    {
                        methodName = match.Groups[1].Value;
                    }
                }
            }

            // Count braces to track method body
            if (methodStartLine != -1)
            {
                braceCount += line.Count(c => c == '{');
                braceCount -= line.Count(c => c == '}');

                // Method end found
                if (braceCount == 0 && line.Contains("}"))
                {
                    var methodLength = i - methodStartLine + 1;
                    if (methodLength > MaxMethodLength)
                    {
                        result.Issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Complexity,
                            Severity = IssueSeverity.Warning,
                            Description = $"Method '{methodName}' is too long ({methodLength} lines)",
                            LineNumber = methodStartLine + 1,
                            ColumnNumber = 1,
                            CodeSegment = lines[methodStartLine],
                            SuggestedFix = $"// Consider refactoring method '{methodName}' into smaller methods"
                        });
                    }

                    methodStartLine = -1;
                    methodName = string.Empty;
                    braceCount = 0;
                }
            }
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Checks for complex conditionals
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="result">Analysis result</param>
    private async Task CheckComplexConditionalsAsync(string[] lines, CodeAnalysisResult result)
    {
        const int MaxConditionalsInLine = 3; // Maximum acceptable number of conditionals in a single line

        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();

            // Count conditional operators
            var andCount = Regex.Matches(line, @"&&").Count;
            var orCount = Regex.Matches(line, @"\|\|").Count;
            var totalConditionals = andCount + orCount;

            if (totalConditionals > MaxConditionalsInLine)
            {
                result.Issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Complexity,
                    Severity = IssueSeverity.Warning,
                    Description = $"Complex conditional with {totalConditionals} operators",
                    LineNumber = i + 1,
                    ColumnNumber = 1,
                    CodeSegment = lines[i],
                    SuggestedFix = "// Consider breaking this condition into multiple smaller conditions or extracting to a method"
                });
            }
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Checks for magic numbers
    /// </summary>
    /// <param name="lines">Lines of code</param>
    /// <param name="result">Analysis result</param>
    private async Task CheckMagicNumbersAsync(string[] lines, CodeAnalysisResult result)
    {
        // Regex to find numeric literals that are not 0, 1, or -1
        var magicNumberRegex = new Regex(@"[^.\w](-?\d+)[^.\w]");

        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();

            // Skip comments, string literals, and constant declarations
            if (line.StartsWith("//") || line.StartsWith("/*") || line.StartsWith("*") || line.Contains("const ") || line.Contains("\""))
            {
                continue;
            }

            var matches = magicNumberRegex.Matches(line);
            foreach (Match match in matches)
            {
                if (match.Groups.Count > 1)
                {
                    var number = match.Groups[1].Value;
                    int value;
                    if (int.TryParse(number, out value) && value != 0 && value != 1 && value != -1)
                    {
                        result.Issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Maintainability,
                            Severity = IssueSeverity.Info,
                            Description = $"Magic number: {number}",
                            LineNumber = i + 1,
                            ColumnNumber = line.IndexOf(number) + 1,
                            CodeSegment = lines[i],
                            SuggestedFix = $"// Consider replacing magic number {number} with a named constant"
                        });
                    }
                }
            }
        }

        await Task.CompletedTask;
    }
}