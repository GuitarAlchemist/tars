using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Defines the pattern definition language for code pattern matching
/// </summary>
public class PatternLanguage
{
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="PatternLanguage"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public PatternLanguage(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Gets the supported pattern languages
    /// </summary>
    public List<string> SupportedLanguages =>
    [
        "Regex", // Regular expression patterns
        "Literal", // Literal string matching
        "AST", // Abstract Syntax Tree patterns
        "Semantic", // Semantic patterns (using semantic analysis)
        "Fuzzy", // Fuzzy matching patterns
        "Template"
    ];

    /// <summary>
    /// Gets the available options for pattern matching
    /// </summary>
    public Dictionary<string, string> AvailableOptions => new()
    {
        { "IgnoreCase", "Whether to ignore case when matching (true/false)" },
        { "IgnoreWhitespace", "Whether to ignore whitespace when matching (true/false)" },
        { "IgnoreComments", "Whether to ignore comments when matching (true/false)" },
        { "MatchWholeWord", "Whether to match whole words only (true/false)" },
        { "MatchExactly", "Whether to match exactly or allow partial matches (true/false)" },
        { "MinConfidence", "Minimum confidence threshold for fuzzy matching (0.0 to 1.0)" },
        { "MaxDistance", "Maximum edit distance for fuzzy matching" },
        { "ContextLines", "Number of context lines to include in matches" },
        { "IncludeNested", "Whether to include nested matches (true/false)" },
        { "MaxMatches", "Maximum number of matches to return" }
    };

    /// <summary>
    /// Validates a pattern definition
    /// </summary>
    /// <param name="pattern">The pattern to validate</param>
    /// <returns>True if the pattern is valid, false otherwise</returns>
    public bool ValidatePattern(CodePattern pattern)
    {
        try
        {
            _logger.LogInformation("Validating pattern: {PatternName}", pattern.Name);

            if (string.IsNullOrWhiteSpace(pattern.Pattern))
            {
                _logger.LogWarning("Pattern {PatternName} has an empty pattern definition", pattern.Name);
                return false;
            }

            switch (pattern.PatternLanguage)
            {
                case "Regex":
                    return ValidateRegexPattern(pattern.Pattern);
                case "Literal":
                    return true; // Literal patterns are always valid
                case "AST":
                    return ValidateAstPattern(pattern.Pattern);
                case "Semantic":
                    return ValidateSemanticPattern(pattern.Pattern);
                case "Fuzzy":
                    return ValidateFuzzyPattern(pattern.Pattern);
                case "Template":
                    return ValidateTemplatePattern(pattern.Pattern);
                default:
                    _logger.LogWarning("Unsupported pattern language: {PatternLanguage}", pattern.PatternLanguage);
                    return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating pattern: {PatternName}", pattern.Name);
            return false;
        }
    }

    /// <summary>
    /// Compiles a pattern into an executable matcher
    /// </summary>
    /// <param name="pattern">The pattern to compile</param>
    /// <returns>The compiled pattern</returns>
    public object? CompilePattern(CodePattern pattern)
    {
        try
        {
            _logger.LogInformation("Compiling pattern: {PatternName}", pattern.Name);

            switch (pattern.PatternLanguage)
            {
                case "Regex":
                    var regexOptions = RegexOptions.Compiled;
                    if (pattern.Options.TryGetValue("IgnoreCase", out var ignoreCaseStr) && bool.TryParse(ignoreCaseStr, out var ignoreCase) && ignoreCase)
                    {
                        regexOptions |= RegexOptions.IgnoreCase;
                    }
                    if (pattern.Options.TryGetValue("IgnoreWhitespace", out var ignoreWhitespaceStr) && bool.TryParse(ignoreWhitespaceStr, out var ignoreWhitespace) && ignoreWhitespace)
                    {
                        regexOptions |= RegexOptions.IgnorePatternWhitespace;
                    }
                    return new Regex(pattern.Pattern, regexOptions);

                case "Literal":
                    return pattern.Pattern;

                case "AST":
                    return CompileAstPattern(pattern.Pattern, pattern.Options);

                case "Semantic":
                    return CompileSemanticPattern(pattern.Pattern, pattern.Options);

                case "Fuzzy":
                    return CompileFuzzyPattern(pattern.Pattern, pattern.Options);

                case "Template":
                    return CompileTemplatePattern(pattern.Pattern, pattern.Options);

                default:
                    _logger.LogWarning("Unsupported pattern language: {PatternLanguage}", pattern.PatternLanguage);
                    return null;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error compiling pattern: {PatternName}", pattern.Name);
            return null;
        }
    }

    /// <summary>
    /// Parses a pattern definition from a string
    /// </summary>
    /// <param name="patternDefinition">The pattern definition string</param>
    /// <returns>The parsed pattern</returns>
    public CodePattern? ParsePatternDefinition(string patternDefinition)
    {
        try
        {
            _logger.LogInformation("Parsing pattern definition");

            // Simple format: name:description:language:pattern[:replacement]
            var parts = patternDefinition.Split(':', 5);
            if (parts.Length < 4)
            {
                _logger.LogWarning("Invalid pattern definition format: {PatternDefinition}", patternDefinition);
                return null;
            }

            var pattern = new CodePattern
            {
                Name = parts[0].Trim(),
                Description = parts[1].Trim(),
                Language = parts[2].Trim(),
                Pattern = parts[3].Trim(),
                PatternLanguage = DeterminePatternLanguage(parts[3].Trim())
            };

            if (parts.Length > 4)
            {
                pattern.Replacement = parts[4].Trim();
            }

            return pattern;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing pattern definition");
            return null;
        }
    }

    /// <summary>
    /// Formats a pattern as a string
    /// </summary>
    /// <param name="pattern">The pattern to format</param>
    /// <returns>The formatted pattern string</returns>
    public string FormatPattern(CodePattern pattern)
    {
        try
        {
            _logger.LogInformation("Formatting pattern: {PatternName}", pattern.Name);

            var result = $"{pattern.Name}:{pattern.Description}:{pattern.Language}:{pattern.Pattern}";
            if (!string.IsNullOrEmpty(pattern.Replacement))
            {
                result += $":{pattern.Replacement}";
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error formatting pattern: {PatternName}", pattern.Name);
            return string.Empty;
        }
    }

    /// <summary>
    /// Determines the pattern language from a pattern string
    /// </summary>
    /// <param name="pattern">The pattern string</param>
    /// <returns>The determined pattern language</returns>
    public string DeterminePatternLanguage(string pattern)
    {
        try
        {
            _logger.LogInformation("Determining pattern language");

            // Check if it's a regex pattern
            if (pattern.StartsWith("/") && pattern.EndsWith("/"))
            {
                return "Regex";
            }

            // Check if it's an AST pattern
            if (pattern.StartsWith("{") && pattern.EndsWith("}"))
            {
                return "AST";
            }

            // Check if it's a template pattern
            if (pattern.Contains("$") || pattern.Contains("${"))
            {
                return "Template";
            }

            // Check if it's a fuzzy pattern
            if (pattern.StartsWith("~"))
            {
                return "Fuzzy";
            }

            // Default to literal
            return "Literal";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error determining pattern language");
            return "Literal";
        }
    }

    private bool ValidateRegexPattern(string pattern)
    {
        try
        {
            // Try to create a regex from the pattern
            _ = new Regex(pattern);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Invalid regex pattern: {Pattern}", pattern);
            return false;
        }
    }

    private bool ValidateAstPattern(string pattern)
    {
        try
        {
            // For now, just check if it's valid JSON
            _ = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(pattern);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Invalid AST pattern: {Pattern}", pattern);
            return false;
        }
    }

    private bool ValidateSemanticPattern(string pattern)
    {
        // Placeholder for semantic pattern validation
        return !string.IsNullOrWhiteSpace(pattern);
    }

    private bool ValidateFuzzyPattern(string pattern)
    {
        // Placeholder for fuzzy pattern validation
        return !string.IsNullOrWhiteSpace(pattern) && pattern.StartsWith("~");
    }

    private bool ValidateTemplatePattern(string pattern)
    {
        try
        {
            // Check if the template has valid placeholders
            var placeholderRegex = new Regex(@"\$\{([^}]+)\}|\$([a-zA-Z0-9_]+)");
            var matches = placeholderRegex.Matches(pattern);
            
            // Ensure all placeholders have valid names
            foreach (Match match in matches)
            {
                var placeholderName = match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value;
                if (string.IsNullOrWhiteSpace(placeholderName))
                {
                    _logger.LogWarning("Invalid placeholder in template pattern: {Pattern}", pattern);
                    return false;
                }
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Invalid template pattern: {Pattern}", pattern);
            return false;
        }
    }

    private object? CompileAstPattern(string pattern, Dictionary<string, string> options)
    {
        // Placeholder for AST pattern compilation
        return pattern;
    }

    private object? CompileSemanticPattern(string pattern, Dictionary<string, string> options)
    {
        // Placeholder for semantic pattern compilation
        return pattern;
    }

    private object? CompileFuzzyPattern(string pattern, Dictionary<string, string> options)
    {
        // Placeholder for fuzzy pattern compilation
        return pattern.TrimStart('~');
    }

    private object? CompileTemplatePattern(string pattern, Dictionary<string, string> options)
    {
        try
        {
            // Extract placeholders
            var placeholderRegex = new Regex(@"\$\{([^}]+)\}|\$([a-zA-Z0-9_]+)");
            var matches = placeholderRegex.Matches(pattern);
            
            var placeholders = new List<string>();
            foreach (Match match in matches)
            {
                var placeholderName = match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value;
                placeholders.Add(placeholderName);
            }
            
            return new { Pattern = pattern, Placeholders = placeholders };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error compiling template pattern: {Pattern}", pattern);
            return null;
        }
    }
}
