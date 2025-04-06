using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for parsing pattern definitions
/// </summary>
public class PatternParser
{
    private readonly ILogger _logger;
    private readonly PatternLanguage _patternLanguage;

    /// <summary>
    /// Initializes a new instance of the <see cref="PatternParser"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="patternLanguage">The pattern language</param>
    public PatternParser(ILogger logger, PatternLanguage patternLanguage)
    {
        _logger = logger;
        _patternLanguage = patternLanguage;
    }

    /// <summary>
    /// Parses a pattern definition file
    /// </summary>
    /// <param name="filePath">The path to the pattern definition file</param>
    /// <returns>The list of parsed patterns</returns>
    public List<CodePattern> ParsePatternFile(string filePath)
    {
        try
        {
            _logger.LogInformation("Parsing pattern file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                _logger.LogWarning("Pattern file not found: {FilePath}", filePath);
                return new List<CodePattern>();
            }

            var fileContent = File.ReadAllText(filePath);
            var fileExtension = Path.GetExtension(filePath).ToLowerInvariant();

            return fileExtension switch
            {
                ".json" => ParseJsonPatternFile(fileContent),
                ".yaml" or ".yml" => ParseYamlPatternFile(fileContent),
                ".meta" => ParseMetaPatternFile(fileContent),
                _ => ParseTextPatternFile(fileContent)
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing pattern file: {FilePath}", filePath);
            return new List<CodePattern>();
        }
    }

    /// <summary>
    /// Parses a pattern definition string
    /// </summary>
    /// <param name="patternDefinition">The pattern definition string</param>
    /// <returns>The parsed pattern</returns>
    public CodePattern? ParsePatternDefinition(string patternDefinition)
    {
        return _patternLanguage.ParsePatternDefinition(patternDefinition);
    }

    /// <summary>
    /// Parses a JSON pattern file
    /// </summary>
    /// <param name="fileContent">The file content</param>
    /// <returns>The list of parsed patterns</returns>
    public List<CodePattern> ParseJsonPatternFile(string fileContent)
    {
        try
        {
            _logger.LogInformation("Parsing JSON pattern file");

            var patterns = JsonSerializer.Deserialize<List<CodePattern>>(fileContent, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return patterns ?? new List<CodePattern>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing JSON pattern file");
            return new List<CodePattern>();
        }
    }

    /// <summary>
    /// Parses a YAML pattern file
    /// </summary>
    /// <param name="fileContent">The file content</param>
    /// <returns>The list of parsed patterns</returns>
    public List<CodePattern> ParseYamlPatternFile(string fileContent)
    {
        try
        {
            _logger.LogInformation("Parsing YAML pattern file");

            // For now, we'll use a simple approach to convert YAML to JSON
            // In a real implementation, we would use a proper YAML parser
            var jsonContent = ConvertYamlToJson(fileContent);
            return ParseJsonPatternFile(jsonContent);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing YAML pattern file");
            return new List<CodePattern>();
        }
    }

    /// <summary>
    /// Parses a meta pattern file
    /// </summary>
    /// <param name="fileContent">The file content</param>
    /// <returns>The list of parsed patterns</returns>
    public List<CodePattern> ParseMetaPatternFile(string fileContent)
    {
        try
        {
            _logger.LogInformation("Parsing meta pattern file");

            var patterns = new List<CodePattern>();
            var ruleRegex = new Regex(@"rule\s+([a-zA-Z0-9_]+)\s*\{([^}]+)\}", RegexOptions.Singleline);
            var ruleMatches = ruleRegex.Matches(fileContent);

            foreach (Match ruleMatch in ruleMatches)
            {
                if (ruleMatch.Groups.Count < 3)
                {
                    continue;
                }

                var ruleName = ruleMatch.Groups[1].Value.Trim();
                var ruleContent = ruleMatch.Groups[2].Value.Trim();

                var pattern = new CodePattern
                {
                    Name = ruleName,
                    Description = ExtractRuleProperty(ruleContent, "description"),
                    Language = ExtractRuleProperty(ruleContent, "language"),
                    Pattern = ExtractRuleProperty(ruleContent, "match"),
                    Replacement = ExtractRuleProperty(ruleContent, "replace"),
                    PatternLanguage = "Template"
                };

                // Parse confidence
                var confidenceStr = ExtractRuleProperty(ruleContent, "confidence");
                if (!string.IsNullOrEmpty(confidenceStr) && double.TryParse(confidenceStr, out var confidence))
                {
                    pattern.ConfidenceThreshold = confidence;
                }

                // Parse tags
                var tagsStr = ExtractRuleProperty(ruleContent, "tags");
                if (!string.IsNullOrEmpty(tagsStr))
                {
                    pattern.Tags = tagsStr.Split(',').Select(t => t.Trim()).ToList();
                }

                patterns.Add(pattern);
            }

            return patterns;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing meta pattern file");
            return new List<CodePattern>();
        }
    }

    /// <summary>
    /// Parses a text pattern file
    /// </summary>
    /// <param name="fileContent">The file content</param>
    /// <returns>The list of parsed patterns</returns>
    public List<CodePattern> ParseTextPatternFile(string fileContent)
    {
        try
        {
            _logger.LogInformation("Parsing text pattern file");

            var patterns = new List<CodePattern>();
            var lines = fileContent.Split('\n');

            foreach (var line in lines)
            {
                var trimmedLine = line.Trim();
                if (string.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("#") || trimmedLine.StartsWith("//"))
                {
                    continue;
                }

                var pattern = _patternLanguage.ParsePatternDefinition(trimmedLine);
                if (pattern != null)
                {
                    patterns.Add(pattern);
                }
            }

            return patterns;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing text pattern file");
            return new List<CodePattern>();
        }
    }

    private string ExtractRuleProperty(string ruleContent, string propertyName)
    {
        var propertyRegex = new Regex($@"{propertyName}\s*:\s*""([^""]+)""|{propertyName}\s*:\s*([^,\r\n]+)", RegexOptions.IgnoreCase);
        var match = propertyRegex.Match(ruleContent);

        if (match.Success)
        {
            return match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value.Trim();
        }

        return string.Empty;
    }

    private string ConvertYamlToJson(string yaml)
    {
        // This is a very simplified YAML to JSON converter
        // In a real implementation, we would use a proper YAML parser
        var lines = yaml.Split('\n');
        var jsonBuilder = new System.Text.StringBuilder();
        jsonBuilder.AppendLine("[");

        var currentPattern = new System.Text.StringBuilder();
        var inPattern = false;

        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            if (string.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("#"))
            {
                continue;
            }

            if (trimmedLine == "---")
            {
                if (inPattern)
                {
                    currentPattern.AppendLine("},");
                    jsonBuilder.Append(currentPattern.ToString());
                    currentPattern.Clear();
                }
                inPattern = true;
                currentPattern.AppendLine("{");
                continue;
            }

            if (inPattern)
            {
                var parts = trimmedLine.Split(':', 2);
                if (parts.Length == 2)
                {
                    var key = parts[0].Trim();
                    var value = parts[1].Trim();

                    if (value.StartsWith("[") && value.EndsWith("]"))
                    {
                        // Array value
                        currentPattern.AppendLine($"  \"{key}\": {value},");
                    }
                    else
                    {
                        // String value
                        currentPattern.AppendLine($"  \"{key}\": \"{value.Replace("\"", "\\\"")}\",");
                    }
                }
            }
        }

        if (inPattern)
        {
            currentPattern.AppendLine("}");
            jsonBuilder.Append(currentPattern.ToString());
        }

        jsonBuilder.AppendLine("]");
        return jsonBuilder.ToString();
    }
}
