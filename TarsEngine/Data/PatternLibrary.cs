using System.Text.Json;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services;

namespace TarsEngine.Data;

/// <summary>
/// Service for managing the pattern library
/// </summary>
public class PatternLibrary
{
    private readonly ILogger _logger;
    private readonly PatternParser _patternParser;
    private readonly string _patternDirectory;
    private readonly Dictionary<string, TarsEngine.Models.CodePattern> _patterns = new();
    private readonly Dictionary<string, List<string>> _patternsByTag = new();
    private readonly Dictionary<string, List<string>> _patternsByLanguage = new();
    private readonly Dictionary<string, List<string>> _patternsByCategory = new();
    private bool _isInitialized = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="PatternLibrary"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="patternParser">The pattern parser</param>
    /// <param name="patternDirectory">The directory containing pattern files</param>
    public PatternLibrary(ILogger logger, PatternParser patternParser, string patternDirectory)
    {
        _logger = logger;
        _patternParser = patternParser;
        _patternDirectory = patternDirectory;
    }

    /// <summary>
    /// Initializes the pattern library
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing pattern library from directory: {PatternDirectory}", _patternDirectory);

            if (!Directory.Exists(_patternDirectory))
            {
                _logger.LogWarning("Pattern directory not found: {PatternDirectory}", _patternDirectory);
                Directory.CreateDirectory(_patternDirectory);
            }

            // Clear existing patterns
            _patterns.Clear();
            _patternsByTag.Clear();
            _patternsByLanguage.Clear();
            _patternsByCategory.Clear();

            // Load patterns from files
            var patternFiles = Directory.GetFiles(_patternDirectory, "*.*", SearchOption.AllDirectories)
                .Where(f => f.EndsWith(".json") || f.EndsWith(".yaml") || f.EndsWith(".yml") || f.EndsWith(".meta") || f.EndsWith(".txt"))
                .ToList();

            _logger.LogInformation("Found {PatternFileCount} pattern files", patternFiles.Count);

            foreach (var file in patternFiles)
            {
                var patterns = _patternParser.ParsePatternFile(file);
                foreach (var pattern in patterns)
                {
                    AddPatternToIndex(pattern);
                }
            }

            _logger.LogInformation("Initialized pattern library with {PatternCount} patterns", _patterns.Count);
            _isInitialized = true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing pattern library");
        }
    }

    /// <summary>
    /// Gets all patterns
    /// </summary>
    /// <param name="language">Optional language filter</param>
    /// <returns>The list of patterns</returns>
    public async Task<List<TarsEngine.Models.CodePattern>> GetPatternsAsync(string? language = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting patterns with language filter: {Language}", language ?? "all");

            if (string.IsNullOrEmpty(language))
            {
                return _patterns.Values.ToList();
            }

            if (_patternsByLanguage.TryGetValue(language.ToLowerInvariant(), out var patternIds))
            {
                return patternIds
                    .Where(id => _patterns.ContainsKey(id))
                    .Select(id => _patterns[id])
                    .ToList();
            }

            return new List<CodePattern>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting patterns");
            return new List<TarsEngine.Models.CodePattern>();
        }
    }

    /// <summary>
    /// Gets a pattern by ID
    /// </summary>
    /// <param name="patternId">The pattern ID</param>
    /// <returns>The pattern, or null if not found</returns>
    public async Task<TarsEngine.Models.CodePattern?> GetPatternAsync(string patternId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting pattern by ID: {PatternId}", patternId);

            if (_patterns.TryGetValue(patternId, out var pattern))
            {
                return pattern;
            }

            _logger.LogWarning("Pattern not found: {PatternId}", patternId);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting pattern: {PatternId}", patternId);
            return null;
        }
    }

    /// <summary>
    /// Gets patterns by tag
    /// </summary>
    /// <param name="tag">The tag</param>
    /// <returns>The list of patterns</returns>
    public async Task<List<TarsEngine.Models.CodePattern>> GetPatternsByTagAsync(string tag)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting patterns by tag: {Tag}", tag);

            if (_patternsByTag.TryGetValue(tag.ToLowerInvariant(), out var patternIds))
            {
                return patternIds
                    .Where(id => _patterns.ContainsKey(id))
                    .Select(id => _patterns[id])
                    .ToList();
            }

            return new List<CodePattern>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting patterns by tag: {Tag}", tag);
            return new List<TarsEngine.Models.CodePattern>();
        }
    }

    /// <summary>
    /// Gets patterns by category
    /// </summary>
    /// <param name="category">The category</param>
    /// <returns>The list of patterns</returns>
    public async Task<List<TarsEngine.Models.CodePattern>> GetPatternsByCategoryAsync(string category)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting patterns by category: {Category}", category);

            if (_patternsByCategory.TryGetValue(category.ToLowerInvariant(), out var patternIds))
            {
                return patternIds
                    .Where(id => _patterns.ContainsKey(id))
                    .Select(id => _patterns[id])
                    .ToList();
            }

            return new List<CodePattern>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting patterns by category: {Category}", category);
            return new List<TarsEngine.Models.CodePattern>();
        }
    }

    /// <summary>
    /// Adds a new pattern
    /// </summary>
    /// <param name="pattern">The pattern to add</param>
    /// <returns>True if the pattern was added successfully, false otherwise</returns>
    public async Task<bool> AddPatternAsync(TarsEngine.Models.CodePattern pattern)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Adding pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);

            if (_patterns.ContainsKey(pattern.Id))
            {
                _logger.LogWarning("Pattern already exists: {PatternId}", pattern.Id);
                return false;
            }

            // Add pattern to index
            AddPatternToIndex(pattern);

            // Save pattern to file
            await SavePatternToFileAsync(pattern);

            _logger.LogInformation("Pattern added: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return false;
        }
    }

    /// <summary>
    /// Updates an existing pattern
    /// </summary>
    /// <param name="pattern">The pattern to update</param>
    /// <returns>True if the pattern was updated successfully, false otherwise</returns>
    public async Task<bool> UpdatePatternAsync(TarsEngine.Models.CodePattern pattern)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Updating pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);

            if (!_patterns.ContainsKey(pattern.Id))
            {
                _logger.LogWarning("Pattern not found: {PatternId}", pattern.Id);
                return false;
            }

            // Remove old pattern from index
            RemovePatternFromIndex(_patterns[pattern.Id]);

            // Update pattern
            pattern.UpdatedAt = DateTime.UtcNow;
            AddPatternToIndex(pattern);

            // Save pattern to file
            await SavePatternToFileAsync(pattern);

            _logger.LogInformation("Pattern updated: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating pattern: {PatternName} ({PatternId})", pattern.Name, pattern.Id);
            return false;
        }
    }

    /// <summary>
    /// Removes a pattern
    /// </summary>
    /// <param name="patternId">The ID of the pattern to remove</param>
    /// <returns>True if the pattern was removed successfully, false otherwise</returns>
    public async Task<bool> RemovePatternAsync(string patternId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Removing pattern: {PatternId}", patternId);

            if (!_patterns.TryGetValue(patternId, out var pattern))
            {
                _logger.LogWarning("Pattern not found: {PatternId}", patternId);
                return false;
            }

            // Remove pattern from index
            RemovePatternFromIndex(pattern);

            // Remove pattern file
            var filePath = GetPatternFilePath(pattern);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }

            _logger.LogInformation("Pattern removed: {PatternId}", patternId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing pattern: {PatternId}", patternId);
            return false;
        }
    }

    /// <summary>
    /// Gets all available tags
    /// </summary>
    /// <returns>The list of tags</returns>
    public async Task<List<string>> GetTagsAsync()
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting all tags");
            return _patternsByTag.Keys.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting tags");
            return new List<string>();
        }
    }

    /// <summary>
    /// Gets all available categories
    /// </summary>
    /// <returns>The list of categories</returns>
    public async Task<List<string>> GetCategoriesAsync()
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting all categories");
            return _patternsByCategory.Keys.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting categories");
            return new List<string>();
        }
    }

    /// <summary>
    /// Gets all available languages
    /// </summary>
    /// <returns>The list of languages</returns>
    public async Task<List<string>> GetLanguagesAsync()
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting all languages");
            return _patternsByLanguage.Keys.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting languages");
            return new List<string>();
        }
    }

    /// <summary>
    /// Ensures the pattern library is initialized
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task EnsureInitializedAsync()
    {
        if (!_isInitialized)
        {
            await InitializeAsync();
        }
    }

    /// <summary>
    /// Adds a pattern to the index
    /// </summary>
    /// <param name="pattern">The pattern to add</param>
    private void AddPatternToIndex(TarsEngine.Models.CodePattern pattern)
    {
        _patterns[pattern.Id] = pattern;

        // Index by language
        var language = pattern.Language.ToLowerInvariant();
        if (!_patternsByLanguage.ContainsKey(language))
        {
            _patternsByLanguage[language] = new List<string>();
        }
        _patternsByLanguage[language].Add(pattern.Id);

        // Index by tags
        foreach (var tag in pattern.Tags)
        {
            var tagKey = tag.ToLowerInvariant();
            if (!_patternsByTag.ContainsKey(tagKey))
            {
                _patternsByTag[tagKey] = new List<string>();
            }
            _patternsByTag[tagKey].Add(pattern.Id);
        }

        // Index by category (from metadata)
        if (pattern.Metadata.TryGetValue("Category", out var category))
        {
            var categoryKey = category.ToLowerInvariant();
            if (!_patternsByCategory.ContainsKey(categoryKey))
            {
                _patternsByCategory[categoryKey] = new List<string>();
            }
            _patternsByCategory[categoryKey].Add(pattern.Id);
        }
    }

    /// <summary>
    /// Removes a pattern from the index
    /// </summary>
    /// <param name="pattern">The pattern to remove</param>
    private void RemovePatternFromIndex(TarsEngine.Models.CodePattern pattern)
    {
        _patterns.Remove(pattern.Id);

        // Remove from language index
        var language = pattern.Language.ToLowerInvariant();
        if (_patternsByLanguage.TryGetValue(language, out var languagePatterns))
        {
            languagePatterns.Remove(pattern.Id);
            if (languagePatterns.Count == 0)
            {
                _patternsByLanguage.Remove(language);
            }
        }

        // Remove from tag index
        foreach (var tag in pattern.Tags)
        {
            var tagKey = tag.ToLowerInvariant();
            if (_patternsByTag.TryGetValue(tagKey, out var tagPatterns))
            {
                tagPatterns.Remove(pattern.Id);
                if (tagPatterns.Count == 0)
                {
                    _patternsByTag.Remove(tagKey);
                }
            }
        }

        // Remove from category index
        if (pattern.Metadata.TryGetValue("Category", out var category))
        {
            var categoryKey = category.ToLowerInvariant();
            if (_patternsByCategory.TryGetValue(categoryKey, out var categoryPatterns))
            {
                categoryPatterns.Remove(pattern.Id);
                if (categoryPatterns.Count == 0)
                {
                    _patternsByCategory.Remove(categoryKey);
                }
            }
        }
    }

    /// <summary>
    /// Saves a pattern to a file
    /// </summary>
    /// <param name="pattern">The pattern to save</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task SavePatternToFileAsync(TarsEngine.Models.CodePattern pattern)
    {
        var filePath = GetPatternFilePath(pattern);
        var directory = Path.GetDirectoryName(filePath);
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory!);
        }

        var json = JsonSerializer.Serialize(pattern, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(filePath, json);
    }

    /// <summary>
    /// Gets the file path for a pattern
    /// </summary>
    /// <param name="pattern">The pattern</param>
    /// <returns>The file path</returns>
    private string GetPatternFilePath(TarsEngine.Models.CodePattern pattern)
    {
        var language = pattern.Language.ToLowerInvariant();
        var category = pattern.Metadata.TryGetValue("Category", out var cat) ? cat.ToLowerInvariant() : "general";
        var fileName = $"{pattern.Id}.json";

        return Path.Combine(_patternDirectory, language, category, fileName);
    }
}
