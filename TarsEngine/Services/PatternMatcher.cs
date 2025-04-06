using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for matching code patterns in source code
/// </summary>
public class PatternMatcher
{
    private readonly ILogger _logger;
    private readonly PatternLanguage _patternLanguage;
    private readonly Dictionary<string, object> _compiledPatterns = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="PatternMatcher"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="patternLanguage">The pattern language</param>
    public PatternMatcher(ILogger logger, PatternLanguage patternLanguage)
    {
        _logger = logger;
        _patternLanguage = patternLanguage;
    }

    /// <summary>
    /// Finds patterns in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="patterns">The patterns to match</param>
    /// <param name="language">The programming language of the code</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    public List<PatternMatch> FindPatterns(string content, List<CodePattern> patterns, string language, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Finding patterns in {Language} code of length {Length}", language, content?.Length ?? 0);

            var matches = new List<PatternMatch>();
            var lines = content.Split('\n');

            // Filter patterns by language
            var applicablePatterns = patterns
                .Where(p => p.Language.Equals(language, StringComparison.OrdinalIgnoreCase) || p.Language.Equals("any", StringComparison.OrdinalIgnoreCase))
                .ToList();

            _logger.LogInformation("Found {PatternCount} applicable patterns for language {Language}", applicablePatterns.Count, language);

            // Apply each pattern
            foreach (var pattern in applicablePatterns)
            {
                var patternMatches = MatchPattern(content, pattern, lines, options);
                matches.AddRange(patternMatches);
            }

            // Sort matches by line number
            matches = matches.OrderBy(m => m.Location.StartLine).ToList();

            _logger.LogInformation("Found {MatchCount} pattern matches in {Language} code", matches.Count, language);
            return matches;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding patterns in {Language} code", language);
            return new List<PatternMatch>();
        }
    }

    /// <summary>
    /// Matches a pattern in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="pattern">The pattern to match</param>
    /// <param name="lines">The lines of the content</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    public List<PatternMatch> MatchPattern(string content, CodePattern pattern, string[] lines, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Matching pattern {PatternName} ({PatternId})", pattern.Name, pattern.Id);

            var matches = new List<PatternMatch>();

            // Get or compile the pattern
            if (!_compiledPatterns.TryGetValue(pattern.Id, out var compiledPattern))
            {
                compiledPattern = _patternLanguage.CompilePattern(pattern);
                if (compiledPattern != null)
                {
                    _compiledPatterns[pattern.Id] = compiledPattern;
                }
            }

            if (compiledPattern == null)
            {
                _logger.LogWarning("Failed to compile pattern {PatternName} ({PatternId})", pattern.Name, pattern.Id);
                return matches;
            }

            // Match based on pattern language
            switch (pattern.PatternLanguage)
            {
                case "Regex":
                    matches.AddRange(MatchRegexPattern(content, pattern, (Regex)compiledPattern, lines, options));
                    break;
                case "Literal":
                    matches.AddRange(MatchLiteralPattern(content, pattern, (string)compiledPattern, lines, options));
                    break;
                case "AST":
                    matches.AddRange(MatchAstPattern(content, pattern, compiledPattern, lines, options));
                    break;
                case "Semantic":
                    matches.AddRange(MatchSemanticPattern(content, pattern, compiledPattern, lines, options));
                    break;
                case "Fuzzy":
                    matches.AddRange(MatchFuzzyPattern(content, pattern, compiledPattern, lines, options));
                    break;
                case "Template":
                    matches.AddRange(MatchTemplatePattern(content, pattern, compiledPattern, lines, options));
                    break;
                default:
                    _logger.LogWarning("Unsupported pattern language: {PatternLanguage}", pattern.PatternLanguage);
                    break;
            }

            _logger.LogInformation("Found {MatchCount} matches for pattern {PatternName} ({PatternId})", matches.Count, pattern.Name, pattern.Id);
            return matches;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error matching pattern {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return new List<PatternMatch>();
        }
    }

    /// <summary>
    /// Matches a regex pattern in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="pattern">The pattern to match</param>
    /// <param name="regex">The compiled regex</param>
    /// <param name="lines">The lines of the content</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    private List<PatternMatch> MatchRegexPattern(string content, CodePattern pattern, Regex regex, string[] lines, Dictionary<string, string>? options = null)
    {
        var matches = new List<PatternMatch>();
        var regexMatches = regex.Matches(content);

        foreach (Match regexMatch in regexMatches)
        {
            // Calculate line and column numbers
            var beforeMatch = content.Substring(0, regexMatch.Index);
            var lineNumber = beforeMatch.Count(c => c == '\n');
            var lastNewline = beforeMatch.LastIndexOf('\n');
            var columnNumber = lastNewline >= 0 ? regexMatch.Index - lastNewline - 1 : regexMatch.Index;

            // Get context lines
            var contextLines = GetContextLines(lines, lineNumber, options);

            // Create pattern match
            var match = new PatternMatch
            {
                PatternId = pattern.Id,
                PatternName = pattern.Name,
                Language = pattern.Language,
                MatchedText = regexMatch.Value,
                Context = string.Join("\n", contextLines),
                Location = new CodeLocation
                {
                    StartLine = lineNumber,
                    EndLine = lineNumber + regexMatch.Value.Count(c => c == '\n'),
                    StartColumn = columnNumber,
                    EndColumn = columnNumber + regexMatch.Length
                },
                Confidence = 1.0, // Regex matches are exact
                SuggestedReplacement = pattern.Replacement,
                ReplacementExplanation = pattern.ReplacementExplanation,
                ExpectedImprovement = pattern.ExpectedImprovement,
                ImpactScore = pattern.ImpactScore,
                DifficultyScore = pattern.DifficultyScore,
                Tags = pattern.Tags
            };

            // Apply replacement if available
            if (!string.IsNullOrEmpty(pattern.Replacement))
            {
                match.SuggestedReplacement = ApplyRegexReplacement(regexMatch, pattern.Replacement);
            }

            matches.Add(match);
        }

        return matches;
    }

    /// <summary>
    /// Matches a literal pattern in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="pattern">The pattern to match</param>
    /// <param name="literalPattern">The literal pattern string</param>
    /// <param name="lines">The lines of the content</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    private List<PatternMatch> MatchLiteralPattern(string content, CodePattern pattern, string literalPattern, string[] lines, Dictionary<string, string>? options = null)
    {
        var matches = new List<PatternMatch>();
        var ignoreCase = options != null && options.TryGetValue("IgnoreCase", out var ignoreCaseStr) && bool.TryParse(ignoreCaseStr, out var ignoreCase_) && ignoreCase_;
        
        var comparison = ignoreCase ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        var index = 0;
        
        while ((index = content.IndexOf(literalPattern, index, comparison)) >= 0)
        {
            // Calculate line and column numbers
            var beforeMatch = content.Substring(0, index);
            var lineNumber = beforeMatch.Count(c => c == '\n');
            var lastNewline = beforeMatch.LastIndexOf('\n');
            var columnNumber = lastNewline >= 0 ? index - lastNewline - 1 : index;

            // Get context lines
            var contextLines = GetContextLines(lines, lineNumber, options);

            // Create pattern match
            var match = new PatternMatch
            {
                PatternId = pattern.Id,
                PatternName = pattern.Name,
                Language = pattern.Language,
                MatchedText = literalPattern,
                Context = string.Join("\n", contextLines),
                Location = new CodeLocation
                {
                    StartLine = lineNumber,
                    EndLine = lineNumber + literalPattern.Count(c => c == '\n'),
                    StartColumn = columnNumber,
                    EndColumn = columnNumber + literalPattern.Length
                },
                Confidence = 1.0, // Literal matches are exact
                SuggestedReplacement = pattern.Replacement,
                ReplacementExplanation = pattern.ReplacementExplanation,
                ExpectedImprovement = pattern.ExpectedImprovement,
                ImpactScore = pattern.ImpactScore,
                DifficultyScore = pattern.DifficultyScore,
                Tags = pattern.Tags
            };

            matches.Add(match);
            index += literalPattern.Length;
        }

        return matches;
    }

    /// <summary>
    /// Matches an AST pattern in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="pattern">The pattern to match</param>
    /// <param name="compiledPattern">The compiled pattern</param>
    /// <param name="lines">The lines of the content</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    private List<PatternMatch> MatchAstPattern(string content, CodePattern pattern, object compiledPattern, string[] lines, Dictionary<string, string>? options = null)
    {
        // Placeholder for AST pattern matching
        // In a real implementation, this would use a syntax tree parser
        _logger.LogInformation("AST pattern matching is not fully implemented yet");
        return new List<PatternMatch>();
    }

    /// <summary>
    /// Matches a semantic pattern in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="pattern">The pattern to match</param>
    /// <param name="compiledPattern">The compiled pattern</param>
    /// <param name="lines">The lines of the content</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    private List<PatternMatch> MatchSemanticPattern(string content, CodePattern pattern, object compiledPattern, string[] lines, Dictionary<string, string>? options = null)
    {
        // Placeholder for semantic pattern matching
        // In a real implementation, this would use semantic analysis
        _logger.LogInformation("Semantic pattern matching is not fully implemented yet");
        return new List<PatternMatch>();
    }

    /// <summary>
    /// Matches a fuzzy pattern in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="pattern">The pattern to match</param>
    /// <param name="compiledPattern">The compiled pattern</param>
    /// <param name="lines">The lines of the content</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    private List<PatternMatch> MatchFuzzyPattern(string content, CodePattern pattern, object compiledPattern, string[] lines, Dictionary<string, string>? options = null)
    {
        // Placeholder for fuzzy pattern matching
        // In a real implementation, this would use fuzzy matching algorithms
        _logger.LogInformation("Fuzzy pattern matching is not fully implemented yet");
        return new List<PatternMatch>();
    }

    /// <summary>
    /// Matches a template pattern in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="pattern">The pattern to match</param>
    /// <param name="compiledPattern">The compiled pattern</param>
    /// <param name="lines">The lines of the content</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    private List<PatternMatch> MatchTemplatePattern(string content, CodePattern pattern, object compiledPattern, string[] lines, Dictionary<string, string>? options = null)
    {
        // Placeholder for template pattern matching
        // In a real implementation, this would use template matching with placeholders
        _logger.LogInformation("Template pattern matching is not fully implemented yet");
        return new List<PatternMatch>();
    }

    /// <summary>
    /// Gets the context lines around a match
    /// </summary>
    /// <param name="lines">The lines of the content</param>
    /// <param name="lineNumber">The line number of the match</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The context lines</returns>
    private string[] GetContextLines(string[] lines, int lineNumber, Dictionary<string, string>? options = null)
    {
        var contextLineCount = 2; // Default context lines before and after
        if (options != null && options.TryGetValue("ContextLines", out var contextLineCountStr) && int.TryParse(contextLineCountStr, out var contextLineCount_))
        {
            contextLineCount = contextLineCount_;
        }

        var startLine = Math.Max(0, lineNumber - contextLineCount);
        var endLine = Math.Min(lines.Length - 1, lineNumber + contextLineCount);
        
        return lines.Skip(startLine).Take(endLine - startLine + 1).ToArray();
    }

    /// <summary>
    /// Applies a regex replacement to a match
    /// </summary>
    /// <param name="match">The regex match</param>
    /// <param name="replacement">The replacement pattern</param>
    /// <returns>The replaced string</returns>
    private string ApplyRegexReplacement(Match match, string replacement)
    {
        try
        {
            // Replace capture group references
            var result = replacement;
            for (int i = 0; i < match.Groups.Count; i++)
            {
                result = result.Replace($"${i}", match.Groups[i].Value);
            }

            // Replace named capture group references
            foreach (var groupName in match.Groups.Keys)
            {
                if (groupName is string name)
                {
                    result = result.Replace($"${name}", match.Groups[name].Value);
                }
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying regex replacement");
            return replacement;
        }
    }

    /// <summary>
    /// Calculates the similarity between two code snippets
    /// </summary>
    /// <param name="source">The source code snippet</param>
    /// <param name="target">The target code snippet</param>
    /// <returns>The similarity score (0.0 to 1.0)</returns>
    public double CalculateSimilarity(string source, string target)
    {
        try
        {
            _logger.LogInformation("Calculating similarity between code snippets");

            if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(target))
            {
                return 0.0;
            }

            // Simple Levenshtein distance-based similarity
            var distance = LevenshteinDistance(source, target);
            var maxLength = Math.Max(source.Length, target.Length);
            var similarity = 1.0 - (double)distance / maxLength;

            return similarity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating similarity");
            return 0.0;
        }
    }

    /// <summary>
    /// Calculates the Levenshtein distance between two strings
    /// </summary>
    /// <param name="s">The first string</param>
    /// <param name="t">The second string</param>
    /// <returns>The Levenshtein distance</returns>
    private int LevenshteinDistance(string s, string t)
    {
        var n = s.Length;
        var m = t.Length;
        var d = new int[n + 1, m + 1];

        if (n == 0)
        {
            return m;
        }

        if (m == 0)
        {
            return n;
        }

        for (int i = 0; i <= n; i++)
        {
            d[i, 0] = i;
        }

        for (int j = 0; j <= m; j++)
        {
            d[0, j] = j;
        }

        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                var cost = (t[j - 1] == s[i - 1]) ? 0 : 1;
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);
            }
        }

        return d[n, m];
    }
}
