using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for fuzzy matching of code patterns
/// </summary>
public class FuzzyMatcher
{
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="FuzzyMatcher"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public FuzzyMatcher(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Finds similar patterns to the provided code
    /// </summary>
    /// <param name="content">The code content to find similar patterns for</param>
    /// <param name="patterns">The patterns to match against</param>
    /// <param name="language">The programming language of the code</param>
    /// <param name="minSimilarity">The minimum similarity score (0.0 to 1.0)</param>
    /// <param name="maxResults">The maximum number of results to return</param>
    /// <returns>The list of similar patterns with their similarity scores</returns>
    public List<(CodePattern Pattern, double Similarity)> FindSimilarPatterns(string content, List<CodePattern> patterns, string language, double minSimilarity = 0.7, int maxResults = 10)
    {
        try
        {
            _logger.LogInformation("Finding similar patterns for {Language} code of length {Length}", language, content?.Length ?? 0);

            var results = new List<(CodePattern Pattern, double Similarity)>();

            // Filter patterns by language
            var applicablePatterns = patterns
                .Where(p => p.Language.Equals(language, StringComparison.OrdinalIgnoreCase) || p.Language.Equals("any", StringComparison.OrdinalIgnoreCase))
                .ToList();

            _logger.LogInformation("Found {PatternCount} applicable patterns for language {Language}", applicablePatterns.Count, language);

            // Calculate similarity for each pattern
            foreach (var pattern in applicablePatterns)
            {
                var similarity = CalculateSimilarity(content, pattern);
                if (similarity >= minSimilarity)
                {
                    results.Add((pattern, similarity));
                }
            }

            // Sort by similarity (descending) and take top results
            results = results
                .OrderByDescending(r => r.Similarity)
                .Take(maxResults)
                .ToList();

            _logger.LogInformation("Found {ResultCount} similar patterns", results.Count);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding similar patterns for {Language} code", language);
            return new List<(CodePattern, double)>();
        }
    }

    /// <summary>
    /// Calculates the similarity between code content and a pattern
    /// </summary>
    /// <param name="content">The code content</param>
    /// <param name="pattern">The pattern</param>
    /// <returns>The similarity score (0.0 to 1.0)</returns>
    public double CalculateSimilarity(string content, CodePattern pattern)
    {
        try
        {
            _logger.LogInformation("Calculating similarity for pattern {PatternName} ({PatternId})", pattern.Name, pattern.Id);

            // Calculate different similarity metrics
            var tokenSimilarity = CalculateTokenSimilarity(content, pattern.Pattern);
            var structuralSimilarity = CalculateStructuralSimilarity(content, pattern.Pattern);
            var semanticSimilarity = CalculateSemanticSimilarity(content, pattern.Pattern);

            // Combine similarity scores with weights
            var combinedSimilarity = (tokenSimilarity * 0.4) + (structuralSimilarity * 0.4) + (semanticSimilarity * 0.2);

            _logger.LogInformation("Similarity for pattern {PatternName} ({PatternId}): {Similarity}", pattern.Name, pattern.Id, combinedSimilarity);
            return combinedSimilarity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating similarity for pattern {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return 0.0;
        }
    }

    /// <summary>
    /// Calculates the token-based similarity between two code snippets
    /// </summary>
    /// <param name="source">The source code snippet</param>
    /// <param name="target">The target code snippet</param>
    /// <returns>The similarity score (0.0 to 1.0)</returns>
    public double CalculateTokenSimilarity(string source, string target)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(target))
            {
                return 0.0;
            }

            // Tokenize the source and target
            var sourceTokens = Tokenize(source);
            var targetTokens = Tokenize(target);

            // Calculate Jaccard similarity
            var intersection = sourceTokens.Intersect(targetTokens).Count();
            var union = sourceTokens.Union(targetTokens).Count();

            return union == 0 ? 0.0 : (double)intersection / union;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating token similarity");
            return 0.0;
        }
    }

    /// <summary>
    /// Calculates the structural similarity between two code snippets
    /// </summary>
    /// <param name="source">The source code snippet</param>
    /// <param name="target">The target code snippet</param>
    /// <returns>The similarity score (0.0 to 1.0)</returns>
    public double CalculateStructuralSimilarity(string source, string target)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(target))
            {
                return 0.0;
            }

            // Extract structural features
            var sourceFeatures = ExtractStructuralFeatures(source);
            var targetFeatures = ExtractStructuralFeatures(target);

            // Calculate feature similarity
            var similarity = CalculateFeatureSimilarity(sourceFeatures, targetFeatures);

            return similarity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating structural similarity");
            return 0.0;
        }
    }

    /// <summary>
    /// Calculates the semantic similarity between two code snippets
    /// </summary>
    /// <param name="source">The source code snippet</param>
    /// <param name="target">The target code snippet</param>
    /// <returns>The similarity score (0.0 to 1.0)</returns>
    public double CalculateSemanticSimilarity(string source, string target)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(source) || string.IsNullOrWhiteSpace(target))
            {
                return 0.0;
            }

            // For now, use a simplified approach based on normalized edit distance
            var distance = LevenshteinDistance(source, target);
            var maxLength = Math.Max(source.Length, target.Length);
            var similarity = 1.0 - (double)distance / maxLength;

            return similarity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating semantic similarity");
            return 0.0;
        }
    }

    /// <summary>
    /// Tokenizes a code snippet
    /// </summary>
    /// <param name="code">The code snippet</param>
    /// <returns>The list of tokens</returns>
    private List<string> Tokenize(string code)
    {
        // Simple tokenization by splitting on whitespace and punctuation
        var tokens = Regex.Split(code, @"[\s\p{P}]")
            .Where(t => !string.IsNullOrWhiteSpace(t))
            .Select(t => t.ToLowerInvariant())
            .ToList();

        return tokens;
    }

    /// <summary>
    /// Extracts structural features from a code snippet
    /// </summary>
    /// <param name="code">The code snippet</param>
    /// <returns>The dictionary of feature counts</returns>
    private Dictionary<string, int> ExtractStructuralFeatures(string code)
    {
        var features = new Dictionary<string, int>();

        // Count basic structural elements
        features["length"] = code.Length;
        features["lines"] = code.Split('\n').Length;
        features["braces"] = code.Count(c => c == '{' || c == '}');
        features["parentheses"] = code.Count(c => c == '(' || c == ')');
        features["brackets"] = code.Count(c => c == '[' || c == ']');
        features["semicolons"] = code.Count(c => c == ';');
        features["commas"] = code.Count(c => c == ',');
        features["dots"] = code.Count(c => c == '.');
        features["equals"] = code.Count(c => c == '=');
        features["operators"] = Regex.Matches(code, @"[+\-*/%&|^!<>]").Count;

        // Count keywords
        var keywords = new[] { "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "return", "try", "catch", "throw", "new", "class", "interface", "enum", "struct", "public", "private", "protected", "internal", "static", "readonly", "const", "virtual", "override", "abstract", "sealed", "async", "await" };
        foreach (var keyword in keywords)
        {
            features[$"kw_{keyword}"] = Regex.Matches(code, $@"\b{keyword}\b").Count;
        }

        return features;
    }

    /// <summary>
    /// Calculates the similarity between two feature dictionaries
    /// </summary>
    /// <param name="features1">The first feature dictionary</param>
    /// <param name="features2">The second feature dictionary</param>
    /// <returns>The similarity score (0.0 to 1.0)</returns>
    private double CalculateFeatureSimilarity(Dictionary<string, int> features1, Dictionary<string, int> features2)
    {
        // Get all feature keys
        var allKeys = features1.Keys.Union(features2.Keys).ToList();

        // Calculate normalized Manhattan distance
        double totalDistance = 0;
        double maxPossibleDistance = 0;

        foreach (var key in allKeys)
        {
            var value1 = features1.TryGetValue(key, out var v1) ? v1 : 0;
            var value2 = features2.TryGetValue(key, out var v2) ? v2 : 0;

            // Normalize values based on the feature type
            double normalizedValue1, normalizedValue2;
            if (key == "length" || key == "lines")
            {
                // For length and lines, use logarithmic normalization
                normalizedValue1 = value1 > 0 ? Math.Log10(value1) : 0;
                normalizedValue2 = value2 > 0 ? Math.Log10(value2) : 0;
                maxPossibleDistance += Math.Max(normalizedValue1, normalizedValue2);
            }
            else
            {
                // For other features, use linear normalization
                var maxValue = Math.Max(value1, value2);
                normalizedValue1 = maxValue > 0 ? (double)value1 / maxValue : 0;
                normalizedValue2 = maxValue > 0 ? (double)value2 / maxValue : 0;
                maxPossibleDistance += 1.0; // Maximum possible distance for normalized values
            }

            totalDistance += Math.Abs(normalizedValue1 - normalizedValue2);
        }

        // Convert distance to similarity
        return maxPossibleDistance > 0 ? 1.0 - (totalDistance / maxPossibleDistance) : 0.0;
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
