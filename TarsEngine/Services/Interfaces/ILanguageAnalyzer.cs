using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for language-specific code analyzers
/// </summary>
public interface ILanguageAnalyzer
{
    /// <summary>
    /// Gets the language supported by this analyzer
    /// </summary>
    string Language { get; }

    /// <summary>
    /// Analyzes code content
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="options">Optional analysis options</param>
    /// <returns>The analysis result</returns>
    Task<CodeAnalysisResult> AnalyzeAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the available analysis options specific to this language
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    Task<Dictionary<string, string>> GetAvailableOptionsAsync();

    /// <summary>
    /// Gets the language-specific issue types
    /// </summary>
    /// <returns>The dictionary of issue types and their descriptions</returns>
    Task<Dictionary<CodeIssueType, string>> GetLanguageSpecificIssueTypesAsync();

    /// <summary>
    /// Gets the language-specific metric types
    /// </summary>
    /// <returns>The dictionary of metric types and their descriptions</returns>
    Task<Dictionary<MetricType, string>> GetLanguageSpecificMetricTypesAsync();
}
