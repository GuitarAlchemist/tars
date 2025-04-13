using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Validates syntax of code changes
/// </summary>
public class SyntaxValidator
{
    private readonly ILogger<SyntaxValidator> _logger;
    private readonly Dictionary<string, Func<string, string, Task<List<ValidationResult>>>> _validators;

    /// <summary>
    /// Initializes a new instance of the <see cref="SyntaxValidator"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public SyntaxValidator(ILogger<SyntaxValidator> logger)
    {
        _logger = logger;
        _validators = new Dictionary<string, Func<string, string, Task<List<ValidationResult>>>>(StringComparer.OrdinalIgnoreCase)
        {
            { ".cs", ValidateCSharpSyntaxAsync },
            { ".fs", ValidateFSharpSyntaxAsync },
            { ".json", ValidateJsonSyntaxAsync },
            { ".xml", ValidateXmlSyntaxAsync },
            { ".csproj", ValidateXmlSyntaxAsync },
            { ".fsproj", ValidateXmlSyntaxAsync },
            { ".sln", ValidateSlnSyntaxAsync }
        };
    }

    /// <summary>
    /// Validates syntax of a file
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <returns>The list of validation results</returns>
    public async Task<List<ValidationResult>> ValidateFileSyntaxAsync(string filePath, string content)
    {
        try
        {
            _logger.LogInformation("Validating syntax of file: {FilePath}", filePath);

            // Get file extension
            var extension = Path.GetExtension(filePath);

            // Check if we have a validator for this file type
            if (_validators.TryGetValue(extension, out var validator))
            {
                return await validator(filePath, content);
            }

            // No validator for this file type, return success
            _logger.LogInformation("No syntax validator available for file type: {Extension}", extension);
            return
            [
                new ValidationResult
                {
                    RuleName = "SyntaxValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = $"No syntax validation available for file type: {extension}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                }
            ];
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating syntax of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "SyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating syntax: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Validates C# syntax
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <returns>The list of validation results</returns>
    private async Task<List<ValidationResult>> ValidateCSharpSyntaxAsync(string filePath, string content)
    {
        try
        {
            _logger.LogInformation("Validating C# syntax of file: {FilePath}", filePath);

            var results = new List<ValidationResult>();

            // Parse the code
            var tree = CSharpSyntaxTree.ParseText(content);
            var root = await tree.GetRootAsync();
            var diagnostics = tree.GetDiagnostics();

            // Check for syntax errors
            var errors = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
            if (errors.Count > 0)
            {
                foreach (var error in errors)
                {
                    var lineSpan = error.Location.GetLineSpan();
                    var line = lineSpan.StartLinePosition.Line + 1;
                    var column = lineSpan.StartLinePosition.Character + 1;

                    results.Add(new ValidationResult
                    {
                        RuleName = "CSharpSyntaxValidation",
                        IsPassed = false,
                        Severity = ValidationRuleSeverity.Error,
                        Message = $"Syntax error at line {line}, column {column}: {error.GetMessage()}",
                        Target = filePath,
                        Timestamp = DateTime.UtcNow,
                        Details = error.ToString(),
                        Metadata = new Dictionary<string, string>
                        {
                            { "Line", line.ToString() },
                            { "Column", column.ToString() },
                            { "ErrorCode", error.Id }
                        }
                    });
                }
            }
            else
            {
                // No syntax errors
                results.Add(new ValidationResult
                {
                    RuleName = "CSharpSyntaxValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = "C# syntax validation passed",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                });
            }

            // Check for warnings
            var warnings = diagnostics.Where(d => d.Severity == DiagnosticSeverity.Warning).ToList();
            foreach (var warning in warnings)
            {
                var lineSpan = warning.Location.GetLineSpan();
                var line = lineSpan.StartLinePosition.Line + 1;
                var column = lineSpan.StartLinePosition.Character + 1;

                results.Add(new ValidationResult
                {
                    RuleName = "CSharpSyntaxValidation",
                    IsPassed = true, // Warnings don't fail validation
                    Severity = ValidationRuleSeverity.Warning,
                    Message = $"Syntax warning at line {line}, column {column}: {warning.GetMessage()}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = warning.ToString(),
                    Metadata = new Dictionary<string, string>
                    {
                        { "Line", line.ToString() },
                        { "Column", column.ToString() },
                        { "WarningCode", warning.Id }
                    }
                });
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating C# syntax of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "CSharpSyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating C# syntax: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Validates F# syntax
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <returns>The list of validation results</returns>
    private async Task<List<ValidationResult>> ValidateFSharpSyntaxAsync(string filePath, string content)
    {
        try
        {
            _logger.LogInformation("Validating F# syntax of file: {FilePath}", filePath);

            // For F#, we'll use a simpler approach since we don't have direct access to the F# compiler
            // In a real implementation, you would use the F# Compiler Services
            var results = new List<ValidationResult>();

            // Check for basic syntax issues
            var issues = await Task.Run(() => CheckFSharpBasicSyntax(content));

            if (issues.Count > 0)
            {
                foreach (var issue in issues)
                {
                    results.Add(new ValidationResult
                    {
                        RuleName = "FSharpSyntaxValidation",
                        IsPassed = false,
                        Severity = ValidationRuleSeverity.Error,
                        Message = issue.Message,
                        Target = filePath,
                        Timestamp = DateTime.UtcNow,
                        Details = issue.Details,
                        Metadata = new Dictionary<string, string>
                        {
                            { "Line", issue.Line.ToString() },
                            { "Column", issue.Column.ToString() }
                        }
                    });
                }
            }
            else
            {
                // No syntax issues found
                results.Add(new ValidationResult
                {
                    RuleName = "FSharpSyntaxValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = "F# syntax validation passed",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                });
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating F# syntax of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "FSharpSyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating F# syntax: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Checks F# code for basic syntax issues
    /// </summary>
    /// <param name="content">The file content</param>
    /// <returns>The list of syntax issues</returns>
    private List<(int Line, int Column, string Message, string Details)> CheckFSharpBasicSyntax(string content)
    {
        var issues = new List<(int Line, int Column, string Message, string Details)>();
        var lines = content.Split('\n');

        // Check for unbalanced parentheses, brackets, and braces
        var parenthesesCount = 0;
        var bracketsCount = 0;
        var bracesCount = 0;
        var lastOpenParenthesis = (Line: 0, Column: 0);
        var lastOpenBracket = (Line: 0, Column: 0);
        var lastOpenBrace = (Line: 0, Column: 0);

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            for (int j = 0; j < line.Length; j++)
            {
                var c = line[j];
                switch (c)
                {
                    case '(':
                        parenthesesCount++;
                        lastOpenParenthesis = (i + 1, j + 1);
                        break;
                    case ')':
                        parenthesesCount--;
                        if (parenthesesCount < 0)
                        {
                            issues.Add((i + 1, j + 1, "Unmatched closing parenthesis", $"Found closing parenthesis without matching opening parenthesis at line {i + 1}, column {j + 1}"));
                            parenthesesCount = 0;
                        }
                        break;
                    case '[':
                        bracketsCount++;
                        lastOpenBracket = (i + 1, j + 1);
                        break;
                    case ']':
                        bracketsCount--;
                        if (bracketsCount < 0)
                        {
                            issues.Add((i + 1, j + 1, "Unmatched closing bracket", $"Found closing bracket without matching opening bracket at line {i + 1}, column {j + 1}"));
                            bracketsCount = 0;
                        }
                        break;
                    case '{':
                        bracesCount++;
                        lastOpenBrace = (i + 1, j + 1);
                        break;
                    case '}':
                        bracesCount--;
                        if (bracesCount < 0)
                        {
                            issues.Add((i + 1, j + 1, "Unmatched closing brace", $"Found closing brace without matching opening brace at line {i + 1}, column {j + 1}"));
                            bracesCount = 0;
                        }
                        break;
                }
            }
        }

        if (parenthesesCount > 0)
        {
            issues.Add((lastOpenParenthesis.Line, lastOpenParenthesis.Column, "Unmatched opening parenthesis", $"Found opening parenthesis without matching closing parenthesis at line {lastOpenParenthesis.Line}, column {lastOpenParenthesis.Column}"));
        }

        if (bracketsCount > 0)
        {
            issues.Add((lastOpenBracket.Line, lastOpenBracket.Column, "Unmatched opening bracket", $"Found opening bracket without matching closing bracket at line {lastOpenBracket.Line}, column {lastOpenBracket.Column}"));
        }

        if (bracesCount > 0)
        {
            issues.Add((lastOpenBrace.Line, lastOpenBrace.Column, "Unmatched opening brace", $"Found opening brace without matching closing brace at line {lastOpenBrace.Line}, column {lastOpenBrace.Column}"));
        }

        // Check for string literal issues
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var inString = false;
            var stringStart = 0;

            for (int j = 0; j < line.Length; j++)
            {
                var c = line[j];
                if (c == '"' && (j == 0 || line[j - 1] != '\\'))
                {
                    if (!inString)
                    {
                        inString = true;
                        stringStart = j;
                    }
                    else
                    {
                        inString = false;
                    }
                }
            }

            if (inString)
            {
                issues.Add((i + 1, stringStart + 1, "Unterminated string literal", $"Found unterminated string literal starting at line {i + 1}, column {stringStart + 1}"));
            }
        }

        return issues;
    }

    /// <summary>
    /// Validates JSON syntax
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <returns>The list of validation results</returns>
    private async Task<List<ValidationResult>> ValidateJsonSyntaxAsync(string filePath, string content)
    {
        try
        {
            _logger.LogInformation("Validating JSON syntax of file: {FilePath}", filePath);

            var results = new List<ValidationResult>();

            try
            {
                // Try to parse the JSON
                await Task.Run(() => System.Text.Json.JsonDocument.Parse(content));

                // No syntax errors
                results.Add(new ValidationResult
                {
                    RuleName = "JsonSyntaxValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = "JSON syntax validation passed",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                });
            }
            catch (System.Text.Json.JsonException ex)
            {
                // Extract line and position information from the exception message
                var match = Regex.Match(ex.Message, @"LineNumber: (\d+) \| BytePositionInLine: (\d+)");
                int line = 1;
                int column = 1;

                if (match.Success)
                {
                    line = int.Parse(match.Groups[1].Value);
                    column = int.Parse(match.Groups[2].Value);
                }

                results.Add(new ValidationResult
                {
                    RuleName = "JsonSyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"JSON syntax error at line {line}, column {column}: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex,
                    Metadata = new Dictionary<string, string>
                    {
                        { "Line", line.ToString() },
                        { "Column", column.ToString() }
                    }
                });
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating JSON syntax of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "JsonSyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating JSON syntax: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Validates XML syntax
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <returns>The list of validation results</returns>
    private async Task<List<ValidationResult>> ValidateXmlSyntaxAsync(string filePath, string content)
    {
        try
        {
            _logger.LogInformation("Validating XML syntax of file: {FilePath}", filePath);

            var results = new List<ValidationResult>();

            try
            {
                // Try to parse the XML
                var doc = new System.Xml.XmlDocument();
                await Task.Run(() => doc.LoadXml(content));

                // No syntax errors
                results.Add(new ValidationResult
                {
                    RuleName = "XmlSyntaxValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = "XML syntax validation passed",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                });
            }
            catch (System.Xml.XmlException ex)
            {
                results.Add(new ValidationResult
                {
                    RuleName = "XmlSyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"XML syntax error at line {ex.LineNumber}, position {ex.LinePosition}: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex,
                    Metadata = new Dictionary<string, string>
                    {
                        { "Line", ex.LineNumber.ToString() },
                        { "Column", ex.LinePosition.ToString() }
                    }
                });
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating XML syntax of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "XmlSyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating XML syntax: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Validates SLN syntax
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <returns>The list of validation results</returns>
    private async Task<List<ValidationResult>> ValidateSlnSyntaxAsync(string filePath, string content)
    {
        try
        {
            _logger.LogInformation("Validating SLN syntax of file: {FilePath}", filePath);

            var results = new List<ValidationResult>();

            // Check for basic SLN syntax issues
            var issues = await Task.Run(() => CheckSlnBasicSyntax(content));

            if (issues.Count > 0)
            {
                foreach (var issue in issues)
                {
                    results.Add(new ValidationResult
                    {
                        RuleName = "SlnSyntaxValidation",
                        IsPassed = false,
                        Severity = ValidationRuleSeverity.Error,
                        Message = issue.Message,
                        Target = filePath,
                        Timestamp = DateTime.UtcNow,
                        Details = issue.Details,
                        Metadata = new Dictionary<string, string>
                        {
                            { "Line", issue.Line.ToString() }
                        }
                    });
                }
            }
            else
            {
                // No syntax issues found
                results.Add(new ValidationResult
                {
                    RuleName = "SlnSyntaxValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = "SLN syntax validation passed",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                });
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating SLN syntax of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "SlnSyntaxValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating SLN syntax: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Checks SLN file for basic syntax issues
    /// </summary>
    /// <param name="content">The file content</param>
    /// <returns>The list of syntax issues</returns>
    private List<(int Line, string Message, string Details)> CheckSlnBasicSyntax(string content)
    {
        var issues = new List<(int Line, string Message, string Details)>();
        var lines = content.Split('\n');

        // Check for required sections
        var hasMicrosoftVisualStudioSolutionFile = false;
        var hasGlobalSection = false;
        var hasEndGlobalSection = false;
        var hasGlobal = false;
        var hasEndGlobal = false;

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();

            if (line.StartsWith("Microsoft Visual Studio Solution File"))
            {
                hasMicrosoftVisualStudioSolutionFile = true;
            }
            else if (line.StartsWith("GlobalSection"))
            {
                hasGlobalSection = true;
            }
            else if (line.StartsWith("EndGlobalSection"))
            {
                hasEndGlobalSection = true;
            }
            else if (line == "Global")
            {
                hasGlobal = true;
            }
            else if (line == "EndGlobal")
            {
                hasEndGlobal = true;
            }
        }

        if (!hasMicrosoftVisualStudioSolutionFile)
        {
            issues.Add((1, "Missing solution file header", "The solution file is missing the 'Microsoft Visual Studio Solution File' header"));
        }

        if (hasGlobalSection && !hasEndGlobalSection)
        {
            issues.Add((1, "Unmatched GlobalSection", "Found GlobalSection without matching EndGlobalSection"));
        }

        if (!hasGlobalSection && hasEndGlobalSection)
        {
            issues.Add((1, "Unmatched EndGlobalSection", "Found EndGlobalSection without matching GlobalSection"));
        }

        if (hasGlobal && !hasEndGlobal)
        {
            issues.Add((1, "Unmatched Global", "Found Global without matching EndGlobal"));
        }

        if (!hasGlobal && hasEndGlobal)
        {
            issues.Add((1, "Unmatched EndGlobal", "Found EndGlobal without matching Global"));
        }

        // Check for project entries
        var projectRegex = new Regex(@"Project\(""\{([^}]+)\}""\) = ""([^""]+)"", ""([^""]+)"", ""\{([^}]+)\}""");
        var projectCount = 0;

        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();

            if (line.StartsWith("Project("))
            {
                projectCount++;
                var match = projectRegex.Match(line);
                if (!match.Success)
                {
                    issues.Add((i + 1, "Invalid project entry", $"Invalid project entry format at line {i + 1}"));
                }
            }
        }

        if (projectCount == 0)
        {
            issues.Add((1, "No project entries", "The solution file does not contain any project entries"));
        }

        return issues;
    }

    /// <summary>
    /// Gets the supported file extensions
    /// </summary>
    /// <returns>The list of supported file extensions</returns>
    public List<string> GetSupportedFileExtensions()
    {
        return _validators.Keys.ToList();
    }
}
