using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the pattern matcher service
/// </summary>
public interface IPatternMatcherService
{
    /// <summary>
    /// Finds patterns in the provided content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="language">The programming language of the code</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    Task<List<PatternMatch>> FindPatternsAsync(string content, string language, Dictionary<string, string>? options = null);

    /// <summary>
    /// Finds patterns in a file
    /// </summary>
    /// <param name="filePath">The path to the file to analyze</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches</returns>
    Task<List<PatternMatch>> FindPatternsInFileAsync(string filePath, Dictionary<string, string>? options = null);

    /// <summary>
    /// Finds patterns in a directory
    /// </summary>
    /// <param name="directoryPath">The path to the directory to analyze</param>
    /// <param name="recursive">Whether to analyze subdirectories</param>
    /// <param name="filePattern">The pattern to match files to analyze</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The list of pattern matches grouped by file</returns>
    Task<Dictionary<string, List<PatternMatch>>> FindPatternsInDirectoryAsync(string directoryPath, bool recursive = true, string filePattern = "*.cs;*.fs", Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets all available patterns
    /// </summary>
    /// <param name="language">Optional language filter</param>
    /// <returns>The list of available patterns</returns>
    Task<List<CodePattern>> GetPatternsAsync(string? language = null);

    /// <summary>
    /// Gets a pattern by ID
    /// </summary>
    /// <param name="patternId">The pattern ID</param>
    /// <returns>The pattern, or null if not found</returns>
    Task<CodePattern?> GetPatternAsync(string patternId);

    /// <summary>
    /// Adds a new pattern
    /// </summary>
    /// <param name="pattern">The pattern to add</param>
    /// <returns>True if the pattern was added successfully, false otherwise</returns>
    Task<bool> AddPatternAsync(CodePattern pattern);

    /// <summary>
    /// Updates an existing pattern
    /// </summary>
    /// <param name="pattern">The pattern to update</param>
    /// <returns>True if the pattern was updated successfully, false otherwise</returns>
    Task<bool> UpdatePatternAsync(CodePattern pattern);

    /// <summary>
    /// Removes a pattern
    /// </summary>
    /// <param name="patternId">The ID of the pattern to remove</param>
    /// <returns>True if the pattern was removed successfully, false otherwise</returns>
    Task<bool> RemovePatternAsync(string patternId);

    /// <summary>
    /// Calculates the similarity between two code snippets
    /// </summary>
    /// <param name="source">The source code snippet</param>
    /// <param name="target">The target code snippet</param>
    /// <param name="language">The programming language of the code</param>
    /// <returns>The similarity score (0.0 to 1.0)</returns>
    Task<double> CalculateSimilarityAsync(string source, string target, string language);

    /// <summary>
    /// Finds similar patterns to the provided code
    /// </summary>
    /// <param name="content">The code content to find similar patterns for</param>
    /// <param name="language">The programming language of the code</param>
    /// <param name="minSimilarity">The minimum similarity score (0.0 to 1.0)</param>
    /// <param name="maxResults">The maximum number of results to return</param>
    /// <returns>The list of similar patterns with their similarity scores</returns>
    Task<List<(CodePattern Pattern, double Similarity)>> FindSimilarPatternsAsync(string content, string language, double minSimilarity = 0.7, int maxResults = 10);

    /// <summary>
    /// Gets the available matching options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    Task<Dictionary<string, string>> GetAvailableOptionsAsync();

    /// <summary>
    /// Gets the supported pattern languages
    /// </summary>
    /// <returns>The list of supported pattern languages</returns>
    Task<List<string>> GetSupportedPatternLanguagesAsync();
}
