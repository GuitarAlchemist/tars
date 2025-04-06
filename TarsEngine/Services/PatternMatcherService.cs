using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Data;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for matching code patterns in source code
/// </summary>
public class PatternMatcherService : IPatternMatcherService
{
    private readonly ILogger<PatternMatcherService> _logger;
    private readonly PatternLanguage _patternLanguage;
    private readonly PatternParser _patternParser;
    private readonly PatternMatcher _patternMatcher;
    private readonly FuzzyMatcher _fuzzyMatcher;
    private readonly PatternLibrary _patternLibrary;
    private readonly CodeAnalyzerService _codeAnalyzerService;

    /// <summary>
    /// Initializes a new instance of the <see cref="PatternMatcherService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="patternLibrary">The pattern library</param>
    /// <param name="codeAnalyzerService">The code analyzer service</param>
    public PatternMatcherService(
        ILogger<PatternMatcherService> logger,
        PatternLibrary patternLibrary,
        CodeAnalyzerService codeAnalyzerService)
    {
        _logger = logger;
        _patternLanguage = new PatternLanguage(logger);
        _patternParser = new PatternParser(logger, _patternLanguage);
        _patternMatcher = new PatternMatcher(logger, _patternLanguage);
        _fuzzyMatcher = new FuzzyMatcher(logger);
        _patternLibrary = patternLibrary;
        _codeAnalyzerService = codeAnalyzerService;
    }

    /// <inheritdoc/>
    public async Task<List<PatternMatch>> FindPatternsAsync(string content, string language, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Finding patterns in {Language} code of length {Length}", language, content?.Length ?? 0);

            // Get patterns from library
            var patterns = await _patternLibrary.GetPatternsAsync(language);
            _logger.LogInformation("Found {PatternCount} patterns for language {Language}", patterns.Count, language);

            // Match patterns
            var matches = _patternMatcher.FindPatterns(content, patterns, language, options);
            _logger.LogInformation("Found {MatchCount} pattern matches", matches.Count);

            return matches;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding patterns in {Language} code", language);
            return new List<PatternMatch>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<PatternMatch>> FindPatternsInFileAsync(string filePath, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Finding patterns in file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                _logger.LogWarning("File not found: {FilePath}", filePath);
                return new List<PatternMatch>();
            }

            // Read file content
            var content = await File.ReadAllTextAsync(filePath);

            // Determine language from file extension
            var extension = Path.GetExtension(filePath).ToLowerInvariant();
            var language = extension switch
            {
                ".cs" => "csharp",
                ".fs" => "fsharp",
                ".js" => "javascript",
                ".ts" => "typescript",
                ".py" => "python",
                ".java" => "java",
                ".cpp" or ".cc" or ".h" or ".hpp" => "cpp",
                ".go" => "go",
                ".rb" => "ruby",
                ".php" => "php",
                ".swift" => "swift",
                ".kt" or ".kts" => "kotlin",
                ".rs" => "rust",
                _ => "unknown"
            };

            // Find patterns
            var matches = await FindPatternsAsync(content, language, options);

            // Set file path in matches
            foreach (var match in matches)
            {
                match.FilePath = filePath;
            }

            return matches;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding patterns in file: {FilePath}", filePath);
            return new List<PatternMatch>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, List<PatternMatch>>> FindPatternsInDirectoryAsync(string directoryPath, bool recursive = true, string filePattern = "*.cs;*.fs", Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Finding patterns in directory: {DirectoryPath}, Recursive: {Recursive}, FilePattern: {FilePattern}", directoryPath, recursive, filePattern);

            if (!Directory.Exists(directoryPath))
            {
                _logger.LogWarning("Directory not found: {DirectoryPath}", directoryPath);
                return new Dictionary<string, List<PatternMatch>>();
            }

            var result = new Dictionary<string, List<PatternMatch>>();
            var filePatterns = filePattern.Split(';');
            var files = new List<string>();

            foreach (var pattern in filePatterns)
            {
                var matchingFiles = Directory.GetFiles(directoryPath, pattern, recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly);
                files.AddRange(matchingFiles);
            }

            _logger.LogInformation("Found {FileCount} files to analyze", files.Count);

            foreach (var file in files)
            {
                var matches = await FindPatternsInFileAsync(file, options);
                if (matches.Count > 0)
                {
                    result[file] = matches;
                }
            }

            _logger.LogInformation("Found patterns in {FileCount} files", result.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding patterns in directory: {DirectoryPath}", directoryPath);
            return new Dictionary<string, List<PatternMatch>>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<CodePattern>> GetPatternsAsync(string? language = null)
    {
        try
        {
            _logger.LogInformation("Getting patterns with language filter: {Language}", language ?? "all");
            return await _patternLibrary.GetPatternsAsync(language);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting patterns");
            return new List<CodePattern>();
        }
    }

    /// <inheritdoc/>
    public async Task<CodePattern?> GetPatternAsync(string patternId)
    {
        try
        {
            _logger.LogInformation("Getting pattern by ID: {PatternId}", patternId);
            return await _patternLibrary.GetPatternAsync(patternId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting pattern: {PatternId}", patternId);
            return null;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> AddPatternAsync(CodePattern pattern)
    {
        try
        {
            _logger.LogInformation("Adding pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return await _patternLibrary.AddPatternAsync(pattern);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> UpdatePatternAsync(CodePattern pattern)
    {
        try
        {
            _logger.LogInformation("Updating pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return await _patternLibrary.UpdatePatternAsync(pattern);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> RemovePatternAsync(string patternId)
    {
        try
        {
            _logger.LogInformation("Removing pattern: {PatternId}", patternId);
            return await _patternLibrary.RemovePatternAsync(patternId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing pattern: {PatternId}", patternId);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<double> CalculateSimilarityAsync(string source, string target, string language)
    {
        try
        {
            _logger.LogInformation("Calculating similarity between code snippets in language: {Language}", language);
            return _fuzzyMatcher.CalculateSimilarity(source, target);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating similarity");
            return 0.0;
        }
    }

    /// <inheritdoc/>
    public async Task<List<(CodePattern Pattern, double Similarity)>> FindSimilarPatternsAsync(string content, string language, double minSimilarity = 0.7, int maxResults = 10)
    {
        try
        {
            _logger.LogInformation("Finding similar patterns for {Language} code of length {Length}", language, content?.Length ?? 0);

            // Get patterns from library
            var patterns = await _patternLibrary.GetPatternsAsync(language);
            _logger.LogInformation("Found {PatternCount} patterns for language {Language}", patterns.Count, language);

            // Find similar patterns
            var similarPatterns = _fuzzyMatcher.FindSimilarPatterns(content, patterns, language, minSimilarity, maxResults);
            _logger.LogInformation("Found {SimilarPatternCount} similar patterns", similarPatterns.Count);

            return similarPatterns;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding similar patterns for {Language} code", language);
            return new List<(CodePattern, double)>();
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        try
        {
            _logger.LogInformation("Getting available options");
            return _patternLanguage.AvailableOptions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available options");
            return new Dictionary<string, string>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<string>> GetSupportedPatternLanguagesAsync()
    {
        try
        {
            _logger.LogInformation("Getting supported pattern languages");
            return _patternLanguage.SupportedLanguages;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting supported pattern languages");
            return new List<string>();
        }
    }
}
